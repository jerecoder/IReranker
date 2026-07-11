#!/usr/bin/env python3
"""Run the fair, live, token-budgeted Robust04 cross-paradigm experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.robust04_cross_paradigm.common import (  # noqa: E402
    MANIFEST_PATH,
    PER_QUERY_DIR,
    RUNS_DIR,
    load_snapshot,
    ndcg_at_k,
    sha256,
    write_csv,
    write_trec_run,
)
from experiments.robust04_cross_paradigm.engine import (  # noqa: E402
    SharedFlanT5Engine,
    SharedSamplingOracle,
    UsageMeter,
)
from experiments.robust04_cross_paradigm.methods import (  # noqa: E402
    render_listwise,
    render_setwise,
    run_listwise_rankgpt,
    run_prp_heapsort,
    run_setwise_heapsort,
)
from ireranker.rankers import get_ranker  # noqa: E402
from ireranker.types import RankingTask  # noqa: E402


def validate_ranking(ranking: list[str], candidates: list[str], condition: str, qid: str) -> None:
    if len(ranking) != len(candidates) or set(ranking) != set(candidates):
        raise RuntimeError(
            f"{condition}/{qid} did not return a full permutation: "
            f"len={len(ranking)}, unique={len(set(ranking))}, expected={len(candidates)}"
        )


def meter_row(meter: UsageMeter) -> dict[str, Any]:
    prompts = meter.directional_prompt_instances
    return {
        "logical_comparisons": meter.logical_comparisons,
        "choice_events": meter.choice_events,
        "prompt_instances": prompts,
        "generation_invocations": meter.generation_invocations,
        "document_instances": meter.document_instances,
        "avg_documents_per_prompt": meter.document_instances / prompts if prompts else 0.0,
        "encoder_nonpad_tokens": meter.encoder_nonpad_tokens,
        "encoder_padded_slots": meter.encoder_padded_slots,
        "decoder_tokens": meter.decoder_tokens,
        "total_model_tokens": meter.total_model_tokens,
        "inference_seconds": meter.inference_seconds,
        "invalid_outputs": meter.invalid_outputs,
        "inconsistent_outputs": meter.inconsistent_outputs,
    }


def completion_matches(
    done_path: Path,
    per_query_path: Path,
    run_path: Path,
    signature: dict[str, Any],
    expected_rows: int,
) -> bool:
    try:
        marker = json.loads(done_path.read_text(encoding="utf-8"))
        return (
            marker.get("status") == "complete"
            and marker.get("signature") == signature
            and int(marker.get("rows", -1)) == expected_rows
            and per_query_path.exists()
            and run_path.exists()
            and marker.get("per_query_sha256") == sha256(per_query_path)
            and marker.get("run_sha256") == sha256(run_path)
        )
    except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError):
        return False


def warm_up_engine(
    engine: SharedFlanT5Engine,
    query_rows: list[dict[str, Any]],
    documents: dict[str, str],
) -> None:
    first = query_rows[0]
    query = engine.truncate_query(str(first["query"]))
    candidate_ids = [str(value) for value in first["candidates"][:4]]
    passages = [engine.truncate_passage(documents[doc_id]) for doc_id in candidate_ids]
    pairwise = [
        engine.render_pairwise(query, passages[0], passages[1]),
        engine.render_pairwise(query, passages[1], passages[0]),
    ]
    engine.generate(
        [pairwise[0]],
        meter=UsageMeter(),
        max_new_tokens=2,
        decoder_prefix=True,
        document_counts=[2],
    )
    engine.generate(
        pairwise,
        meter=UsageMeter(),
        max_new_tokens=2,
        decoder_prefix=True,
        document_counts=[2, 2],
    )
    engine.generate(
        [render_setwise(query, passages[:3])],
        meter=UsageMeter(),
        max_new_tokens=2,
        decoder_prefix=True,
        document_counts=[3],
    )
    engine.generate(
        [render_listwise(query, passages)],
        meter=UsageMeter(),
        max_new_tokens=20,
        decoder_prefix=False,
        document_counts=[4],
    )
    print("Completed unmeasured in-process warm-up for all prompt batch shapes")


def load_partial_condition(
    *,
    inprogress_path: Path,
    per_query_path: Path,
    run_path: Path,
    signature: dict[str, Any],
    selected: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[tuple[str, list[str]]]] | None:
    """Load a structurally valid contiguous prefix, or return None to restart it."""
    try:
        marker = json.loads(inprogress_path.read_text(encoding="utf-8"))
        if marker != {"status": "in_progress", "signature": signature}:
            return None
        with per_query_path.open(newline="", encoding="utf-8") as handle:
            results = [dict(row) for row in csv.DictReader(handle)]
        if not results or len(results) > len(selected):
            return None

        run_groups: list[tuple[str, list[str]]] = []
        with run_path.open(encoding="utf-8") as handle:
            current_qid: str | None = None
            current_docs: list[str] = []
            for line in handle:
                parts = line.split()
                if len(parts) < 6:
                    return None
                qid, doc_id = parts[0], parts[2]
                if current_qid is not None and qid != current_qid:
                    run_groups.append((current_qid, current_docs))
                    current_docs = []
                current_qid = qid
                if int(parts[3]) != len(current_docs) + 1:
                    return None
                current_docs.append(doc_id)
            if current_qid is not None:
                run_groups.append((current_qid, current_docs))
        if len(run_groups) != len(results):
            return None

        expected_qids = [str(row["query_id"]) for row in selected[: len(results)]]
        result_qids = [str(row.get("query_id", "")) for row in results]
        run_qids = [qid for qid, _ in run_groups]
        if result_qids != expected_qids or run_qids != expected_qids:
            return None
        for source, (qid, ranking) in zip(selected, run_groups):
            candidates = [str(value) for value in source["candidates"]]
            validate_ranking(ranking, candidates, str(signature["condition"]), qid)
        return results, run_groups
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError, ValueError):
        return None


def run_condition(
    *,
    condition: str,
    method: str,
    variant: str,
    token_budget: int | None,
    seed: int,
    query_rows: list[dict[str, Any]],
    documents: dict[str, str],
    engine: SharedFlanT5Engine | None,
    experiment_signature: dict[str, Any],
    max_queries: int | None,
    resume: bool,
) -> None:
    per_query_path = PER_QUERY_DIR / f"{condition}.csv"
    run_path = RUNS_DIR / f"{condition}.txt"
    done_path = PER_QUERY_DIR / f"{condition}.done"
    inprogress_path = PER_QUERY_DIR / f"{condition}.inprogress.json"
    selected = query_rows[:max_queries] if max_queries else query_rows
    query_ids = [str(row["query_id"]) for row in selected]
    signature = {
        **experiment_signature,
        "condition": condition,
        "method": method,
        "variant": variant,
        "token_budget": token_budget,
        "seed": seed,
        "query_count": len(selected),
        "query_ids_sha256": hashlib.sha256("\n".join(query_ids).encode()).hexdigest(),
    }
    if resume and completion_matches(
        done_path, per_query_path, run_path, signature, len(selected)
    ):
        print(f"SKIP completed: {condition}")
        return
    if resume and done_path.exists():
        print(f"RERUN stale or incompatible completion marker: {condition}")
    done_path.unlink(missing_ok=True)
    partial = (
        load_partial_condition(
            inprogress_path=inprogress_path,
            per_query_path=per_query_path,
            run_path=run_path,
            signature=signature,
            selected=selected,
        )
        if resume
        else None
    )
    if partial is None:
        results: list[dict[str, Any]] = []
        rankings: list[tuple[str, list[str]]] = []
        per_query_path.unlink(missing_ok=True)
        run_path.unlink(missing_ok=True)
        inprogress_path.parent.mkdir(parents=True, exist_ok=True)
        inprogress_path.write_text(
            json.dumps({"status": "in_progress", "signature": signature}, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        results, rankings = partial
        print(f"CONTINUE partial condition: {condition} at query {len(results) + 1}")
    torch = engine.torch if engine is not None else None

    oracle = None
    ranker = None
    if method == "mohajer":
        assert engine is not None
        oracle = SharedSamplingOracle(
            engine=engine,
            queries={str(row["query_id"]): str(row["query"]) for row in selected},
            documents=documents,
            seed=seed,
            token_limit=token_budget,
        )
        ranker = get_ranker("mohajer (ir)", oracle=oracle, seed=seed, top_k=10)
        ranker.set_dataset("robust04", split="test", query_ids=[str(row["query_id"]) for row in selected])

    completed_queries = len(results)
    for index, row in enumerate(selected[completed_queries:], start=completed_queries + 1):
        qid = str(row["query_id"])
        query = str(row["query"])
        candidates = [str(value) for value in row["candidates"]]
        qrels = {str(key): int(value) for key, value in row["qrels"].items()}
        meter = UsageMeter(token_limit=token_budget)

        if torch is not None and engine is not None and engine.device_type == "cuda":
            torch.cuda.reset_peak_memory_stats(engine.device)
            torch.cuda.synchronize(engine.device)
        started = time.perf_counter()

        if method == "bm25":
            ranking = list(candidates)
        elif method == "mohajer":
            assert oracle is not None and ranker is not None
            task = RankingTask(
                query_id=qid,
                candidate_ids=list(candidates),
                y_true=[float(qrels.get(doc_id, 0)) for doc_id in candidates],
                dataset_path=str(ROOT / "data/external/beir/robust04"),
            )
            indices = ranker.rank(task)
            ranking = [candidates[position] for position in indices]
            meter = oracle.meter
        elif method == "prp":
            assert engine is not None
            ranking = run_prp_heapsort(
                query, candidates, documents, engine=engine, meter=meter, k=10
            )
        elif method == "setwise":
            assert engine is not None
            ranking = run_setwise_heapsort(
                query, candidates, documents, engine=engine, meter=meter, num_child=2, k=10
            )
        elif method == "listwise":
            assert engine is not None
            ranking = run_listwise_rankgpt(
                query,
                candidates,
                documents,
                engine=engine,
                meter=meter,
                window_size=4,
                step_size=2,
                repeats=5,
            )
        else:
            raise ValueError(method)

        if torch is not None and engine is not None and engine.device_type == "cuda":
            torch.cuda.synchronize(engine.device)
        wall_seconds = time.perf_counter() - started
        peak_memory = (
            int(torch.cuda.max_memory_allocated(engine.device))
            if torch is not None and engine is not None and engine.device_type == "cuda"
            else 0
        )
        validate_ranking(ranking, candidates, condition, qid)
        rankings.append((qid, ranking))
        results.append({
            "dataset": "robust04",
            "condition": condition,
            "method": method,
            "variant": variant,
            "token_budget": token_budget if token_budget is not None else "",
            "seed": seed,
            "query_id": qid,
            "ndcg10": ndcg_at_k(ranking, qrels, 10),
            **meter_row(meter),
            "query_wall_seconds": wall_seconds,
            "peak_gpu_memory_bytes": peak_memory,
        })
        write_csv(per_query_path, results)
        write_trec_run(run_path, rankings, condition)
        print(
            f"{condition}: {index}/{len(selected)} qid={qid} "
            f"ndcg10={results[-1]['ndcg10']:.4f} tokens={meter.total_model_tokens}"
        )
    marker = {
        "status": "complete",
        "signature": signature,
        "rows": len(results),
        "per_query_sha256": sha256(per_query_path),
        "run_sha256": sha256(run_path),
    }
    done_path.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    inprogress_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["bm25", "mohajer", "prp", "setwise", "listwise"],
        default=["bm25", "mohajer", "prp", "setwise", "listwise"],
    )
    parser.add_argument("--token-budgets", type=int, nargs="+", default=[25000, 50000, 75000, 100000, 125000])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--model", default="google/flan-t5-large")
    parser.add_argument(
        "--model-revision", default="0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--query-tokens", type=int, default=32)
    parser.add_argument("--passage-tokens", type=int, default=100)
    parser.add_argument("--encoder-max-tokens", type=int, default=768)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if any(value <= 0 for value in args.token_budgets) or len(set(args.token_budgets)) != len(args.token_budgets):
        parser.error("--token-budgets must contain unique positive integers")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not contain duplicates")
    if args.max_queries is not None and args.max_queries <= 0:
        parser.error("--max-queries must be positive")
    if min(args.query_tokens, args.passage_tokens, args.encoder_max_tokens) <= 0:
        parser.error("token limits must be positive")

    query_rows, documents = load_snapshot()
    needs_model = any(method != "bm25" for method in args.methods)
    engine = (
        SharedFlanT5Engine(
            model_name=args.model,
            model_revision=args.model_revision,
            device=args.device,
            encoder_max_tokens=args.encoder_max_tokens,
            query_tokens=args.query_tokens,
            passage_tokens=args.passage_tokens,
        )
        if needs_model
        else None
    )
    if engine is not None:
        warm_up_engine(engine, query_rows, documents)
    source_paths = [
        Path(__file__),
        ROOT / "experiments/robust04_cross_paradigm/common.py",
        ROOT / "experiments/robust04_cross_paradigm/engine.py",
        ROOT / "experiments/robust04_cross_paradigm/methods.py",
        ROOT / "ireranker/rankers/mohajer_ranker.py",
        ROOT / "ireranker/data/public_tasks.py",
    ]
    experiment_signature = {
        "protocol_version": 1,
        "dataset": "robust04",
        "snapshot_manifest_sha256": sha256(MANIFEST_PATH),
        "model": args.model,
        "model_revision": args.model_revision,
        "device": args.device,
        "gpu": (
            engine.torch.cuda.get_device_name(engine.device)
            if engine is not None and engine.device_type == "cuda"
            else args.device
        ),
        "torch": str(engine.torch.__version__) if engine is not None else "not-loaded",
        "cuda": str(engine.torch.version.cuda) if engine is not None else "not-loaded",
        "transformers": (
            importlib.metadata.version("transformers") if engine is not None else "not-loaded"
        ),
        "sentencepiece": (
            importlib.metadata.version("sentencepiece") if engine is not None else "not-loaded"
        ),
        "python": sys.version,
        "query_tokens": args.query_tokens,
        "passage_tokens": args.passage_tokens,
        "encoder_max_tokens": args.encoder_max_tokens,
        "model_output_cache": False,
        "unmeasured_prompt_shape_warmup": True,
        "source_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
            for path in source_paths
        },
    }

    if "bm25" in args.methods:
        run_condition(
            condition="bm25",
            method="bm25",
            variant="top100",
            token_budget=None,
            seed=42,
            query_rows=query_rows,
            documents=documents,
            engine=None,
            experiment_signature=experiment_signature,
            max_queries=args.max_queries,
            resume=args.resume,
        )

    for method in ("mohajer", "prp", "setwise", "listwise"):
        if method not in args.methods:
            continue
        seeds = args.seeds if method == "mohajer" else [42]
        variant = {
            "mohajer": "sampling_shared_prp_prompt",
            "prp": "bidirectional_heapsort",
            "setwise": "heapsort_c3",
            "listwise": "rankgpt_w4_s2_r5",
        }[method]
        for token_budget in args.token_budgets:
            for seed in seeds:
                condition = f"{method}_{variant}_t{token_budget}_s{seed}"
                run_condition(
                    condition=condition,
                    method=method,
                    variant=variant,
                    token_budget=token_budget,
                    seed=seed,
                    query_rows=query_rows,
                    documents=documents,
                    engine=engine,
                    experiment_signature=experiment_signature,
                    max_queries=args.max_queries,
                    resume=args.resume,
                )


if __name__ == "__main__":
    main()
