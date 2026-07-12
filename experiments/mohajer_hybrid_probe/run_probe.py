#!/usr/bin/env python3
"""Run the ordered, early-stoppable Mohajer hybrid screen with live FLAN-T5."""

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

from experiments.mohajer_hybrid_probe.common import (  # noqa: E402
    DATASET_ORDER,
    EXP_DIR,
    HYBRID_STAGE_A_FRACTION,
    METHOD_ORDER,
    MOHAJER_FAMILY,
    PER_QUERY_DIR,
    RESULTS_DIR,
    RUNS_DIR,
    load_snapshot,
    mean,
    ndcg_at_k,
    pareto_methods,
    quality_gate_methods,
    sha256,
    snapshot_dir,
    strong_mohajer_failure,
    write_csv,
    write_trec_run,
)
from experiments.mohajer_hybrid_probe.engine import (  # noqa: E402
    SharedFlanT5Engine,
    UsageMeter,
)
from experiments.mohajer_hybrid_probe.methods import (  # noqa: E402
    MethodResult,
    run_bubble,
    run_mohajer,
    run_mohajer_bubble,
    run_mohajer_hybrid,
    run_standalone,
)
from experiments.robust04_cross_paradigm.methods import (  # noqa: E402
    render_listwise,
    render_setwise,
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


def validate_ranking(ranking: list[str], candidates: list[str], label: str) -> None:
    if len(ranking) != len(candidates) or set(ranking) != set(candidates):
        raise RuntimeError(f"{label} did not return a full candidate permutation")


def completion_matches(
    done_path: Path,
    csv_path: Path,
    run_path: Path,
    signature: dict[str, Any],
    expected_rows: int,
) -> bool:
    try:
        marker = json.loads(done_path.read_text(encoding="utf-8"))
        return (
            marker["status"] == "complete"
            and marker["signature"] == signature
            and int(marker["rows"]) == expected_rows
            and marker["csv_sha256"] == sha256(csv_path)
            and marker["run_sha256"] == sha256(run_path)
        )
    except (FileNotFoundError, KeyError, json.JSONDecodeError, TypeError, ValueError):
        return False


def execute_method(
    method: str,
    *,
    row: dict[str, Any],
    documents: dict[str, str],
    engine: SharedFlanT5Engine,
    seed: int,
    token_budget: int,
) -> MethodResult:
    if method == "mohajer":
        return run_mohajer(
            row=row, documents=documents, engine=engine, seed=seed,
            token_budget=token_budget,
        )
    if method == "mohajer_bubble":
        return run_mohajer_bubble(
            row=row, documents=documents, engine=engine, seed=seed,
            token_budget=token_budget,
        )
    if method == "mohajer_setwise":
        return run_mohajer_hybrid(
            "setwise", row=row, documents=documents, engine=engine, seed=seed,
            token_budget=token_budget,
        )
    if method == "mohajer_listwise":
        return run_mohajer_hybrid(
            "listwise", row=row, documents=documents, engine=engine, seed=seed,
            token_budget=token_budget,
        )
    if method in {"setwise", "listwise"}:
        return run_standalone(
            method, row=row, documents=documents, engine=engine, seed=seed,
            token_budget=token_budget,
        )
    if method == "bubble":
        return run_bubble(
            row=row, documents=documents, engine=engine, seed=seed,
            token_budget=token_budget,
        )
    raise ValueError(method)


def run_condition(
    *,
    dataset: str,
    method: str,
    budget: int | None,
    seed: int,
    queries: list[dict[str, Any]],
    documents: dict[str, str],
    engine: SharedFlanT5Engine | None,
    base_signature: dict[str, Any],
    resume: bool,
) -> list[dict[str, Any]]:
    condition = "bm25" if method == "bm25" else f"{method}_t{budget}_s{seed}"
    csv_path = PER_QUERY_DIR / dataset / f"{condition}.csv"
    run_path = RUNS_DIR / dataset / f"{condition}.txt"
    done_path = PER_QUERY_DIR / dataset / f"{condition}.done"
    qids = [str(row["query_id"]) for row in queries]
    signature = {
        **base_signature,
        "dataset": dataset,
        "method": method,
        "budget": budget,
        "seed": seed,
        "condition": condition,
        "query_ids_sha256": hashlib.sha256("\n".join(qids).encode()).hexdigest(),
    }
    if resume and completion_matches(done_path, csv_path, run_path, signature, len(queries)):
        print(f"SKIP completed: {dataset}/{condition}")
        with csv_path.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    done_path.unlink(missing_ok=True)

    result_rows: list[dict[str, Any]] = []
    rankings: list[tuple[str, list[str]]] = []
    for index, row in enumerate(queries, start=1):
        qid = str(row["query_id"])
        candidates = [str(value) for value in row["candidates"]]
        qrels = {str(key): int(value) for key, value in row["qrels"].items()}
        torch = engine.torch if engine is not None else None
        if engine is not None and engine.device_type == "cuda":
            torch.cuda.reset_peak_memory_stats(engine.device)
            torch.cuda.synchronize(engine.device)
        started = time.perf_counter()

        if method == "bm25":
            ranking = list(candidates)
            meter = UsageMeter()
            stage_a_tokens = 0
            stage_b_tokens = 0
        else:
            if engine is None or budget is None:
                raise RuntimeError(f"{method} requires an engine and token budget")
            output = execute_method(
                method,
                row=row,
                documents=documents,
                engine=engine,
                seed=seed,
                token_budget=budget,
            )
            ranking = output.ranking
            meter = output.meter
            stage_a_tokens = output.stage_a_tokens
            stage_b_tokens = output.stage_b_tokens

        if engine is not None and engine.device_type == "cuda":
            torch.cuda.synchronize(engine.device)
        wall_seconds = time.perf_counter() - started
        peak_memory = (
            int(torch.cuda.max_memory_allocated(engine.device))
            if engine is not None and engine.device_type == "cuda"
            else 0
        )
        validate_ranking(ranking, candidates, f"{dataset}/{condition}/{qid}")
        rankings.append((qid, ranking))
        result_rows.append({
            "dataset": dataset,
            "condition": condition,
            "method": method,
            "token_budget": budget if budget is not None else "",
            "seed": seed,
            "query_id": qid,
            "ndcg10": ndcg_at_k(ranking, qrels, 10),
            "stage_a_tokens": stage_a_tokens,
            "stage_b_tokens": stage_b_tokens,
            **meter_row(meter),
            "query_wall_seconds": wall_seconds,
            "peak_gpu_memory_bytes": peak_memory,
        })
        write_csv(csv_path, result_rows)
        write_trec_run(run_path, rankings, condition)
        print(
            f"{dataset}/{condition}: {index}/{len(queries)} qid={qid} "
            f"ndcg10={result_rows[-1]['ndcg10']:.4f} tokens={meter.total_model_tokens}"
        )

    marker = {
        "status": "complete",
        "signature": signature,
        "rows": len(result_rows),
        "csv_sha256": sha256(csv_path),
        "run_sha256": sha256(run_path),
    }
    done_path.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    return result_rows


def rows_by_query(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {str(row["query_id"]): float(row["ndcg10"]) for row in rows}


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "ndcg10": mean(float(row["ndcg10"]) for row in rows),
        "avg_tokens": mean(float(row["total_model_tokens"]) for row in rows),
    }


def qualifying_method(
    completed: dict[str, list[dict[str, Any]]], *, budget: int
) -> str | None:
    bm25 = rows_by_query(completed["bm25"])
    summaries: list[dict[str, Any]] = []
    for method in METHOD_ORDER[1:]:
        key = f"{method}:{budget}"
        if key not in completed:
            continue
        summary = {"method": method, **summarize(completed[key])}
        summaries.append(summary)
    frontier = pareto_methods(
        [{"method": "bm25", "ndcg10": mean(bm25.values()), "avg_tokens": 0.0}]
        + summaries
    )
    candidates: list[tuple[float, str]] = []
    bm_mean = mean(bm25.values())
    for method in MOHAJER_FAMILY:
        key = f"{method}:{budget}"
        if key not in completed:
            continue
        values = rows_by_query(completed[key])
        qids = sorted(set(bm25) & set(values))
        delta = mean(values[qid] - bm25[qid] for qid in qids)
        wins = sum(values[qid] > bm25[qid] for qid in qids)
        if delta >= 0.02 and wins >= 2 and method in frontier:
            candidates.append((mean(values.values()) - bm_mean, method))
    return max(candidates)[1] if candidates else None


def warm_up(engine: SharedFlanT5Engine, row: dict[str, Any], documents: dict[str, str]) -> None:
    query = engine.truncate_query(str(row["query"]))
    ids = [str(value) for value in row["candidates"][:4]]
    passages = [engine.truncate_passage(documents[doc_id]) for doc_id in ids]
    prompts = [
        engine.render_pairwise(query, passages[0], passages[1]),
        engine.render_pairwise(query, passages[1], passages[0]),
    ]
    engine.generate(
        prompts, meter=UsageMeter(), max_new_tokens=2, decoder_prefix=True,
        document_counts=[2, 2],
    )
    engine.generate(
        [render_setwise(query, passages[:3])], meter=UsageMeter(), max_new_tokens=2,
        decoder_prefix=True, document_counts=[3],
    )
    engine.generate(
        [render_listwise(query, passages)], meter=UsageMeter(), max_new_tokens=20,
        decoder_prefix=False, document_counts=[4],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASET_ORDER, default=DATASET_ORDER)
    parser.add_argument("--budgets", type=int, nargs="+", default=[100000, 50000])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="google/flan-t5-large")
    parser.add_argument(
        "--model-revision", default="0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--query-tokens", type=int, default=32)
    parser.add_argument("--passage-tokens", type=int, default=100)
    parser.add_argument("--encoder-max-tokens", type=int, default=768)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-early-stop", action="store_true")
    args = parser.parse_args()
    if len(args.budgets) < 2 or any(value <= 0 for value in args.budgets):
        parser.error("provide at least two positive token budgets, active budget first")

    snapshots = {dataset: load_snapshot(dataset) for dataset in args.datasets}
    engine = SharedFlanT5Engine(
        model_name=args.model,
        model_revision=args.model_revision,
        device=args.device,
        query_tokens=args.query_tokens,
        passage_tokens=args.passage_tokens,
        encoder_max_tokens=args.encoder_max_tokens,
    )
    first_dataset = args.datasets[0]
    warm_up(engine, snapshots[first_dataset][0][0], snapshots[first_dataset][1])

    source_paths = [
        Path(__file__),
        EXP_DIR / "common.py",
        EXP_DIR / "engine.py",
        EXP_DIR / "methods.py",
        ROOT / "experiments/robust04_cross_paradigm/engine.py",
        ROOT / "experiments/robust04_cross_paradigm/methods.py",
        ROOT / "ireranker/rankers/mohajer_ranker.py",
        ROOT / "ireranker/rankers/mohajer_bubble_ranker.py",
        ROOT / "ireranker/rankers/bubble_ranker.py",
    ]
    common_signature = {
        "protocol_version": 2,
        "model": args.model,
        "model_revision": args.model_revision,
        "device": args.device,
        "gpu": (
            engine.torch.cuda.get_device_name(engine.device)
            if engine.device_type == "cuda" else args.device
        ),
        "torch": str(engine.torch.__version__),
        "cuda": str(engine.torch.version.cuda),
        "transformers": importlib.metadata.version("transformers"),
        "sentencepiece": importlib.metadata.version("sentencepiece"),
        "python": sys.version,
        "query_tokens": args.query_tokens,
        "passage_tokens": args.passage_tokens,
        "encoder_max_tokens": args.encoder_max_tokens,
        "model_output_cache": False,
        "hybrid_stage_a_fraction": HYBRID_STAGE_A_FRACTION,
        "source_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
            for path in source_paths
        },
    }
    decisions: list[dict[str, Any]] = []
    stop_globally = False
    active_budget = args.budgets[0]

    for dataset in args.datasets:
        queries, documents, manifest = snapshots[dataset]
        signature = {
            **common_signature,
            "snapshot_manifest_sha256": sha256(snapshot_dir(dataset) / "manifest.json"),
        }
        completed: dict[str, list[dict[str, Any]]] = {}
        completed["bm25"] = run_condition(
            dataset=dataset, method="bm25", budget=None, seed=args.seed,
            queries=queries, documents=documents, engine=None,
            base_signature=signature, resume=args.resume,
        )
        for method in ("mohajer", "mohajer_bubble"):
            completed[f"{method}:{active_budget}"] = run_condition(
                dataset=dataset, method=method, budget=active_budget, seed=args.seed,
                queries=queries, documents=documents, engine=engine,
                base_signature=signature, resume=args.resume,
            )

        failed_gate = strong_mohajer_failure(
            rows_by_query(completed["bm25"]),
            rows_by_query(completed[f"mohajer:{active_budget}"]),
            rows_by_query(completed[f"mohajer_bubble:{active_budget}"]),
        )
        if failed_gate and not args.no_early_stop:
            decision = {
                "dataset": dataset,
                "decision": "reject_after_mohajer_gate",
                "active_budget": active_budget,
            }
            decisions.append(decision)
            print(f"EARLY STOP dataset: {json.dumps(decision)}")
            continue

        for method in ("mohajer_setwise", "mohajer_listwise"):
            completed[f"{method}:{active_budget}"] = run_condition(
                dataset=dataset, method=method, budget=active_budget, seed=args.seed,
                queries=queries, documents=documents, engine=engine,
                base_signature=signature, resume=args.resume,
            )

        quality_candidates = quality_gate_methods(
            rows_by_query(completed["bm25"]),
            {
                method: rows_by_query(completed[f"{method}:{active_budget}"])
                for method in MOHAJER_FAMILY
            },
        )
        if not quality_candidates and not args.no_early_stop:
            decision = {
                "dataset": dataset,
                "decision": "reject_after_hybrid_quality_gate",
                "active_budget": active_budget,
            }
            decisions.append(decision)
            print(f"EARLY STOP dataset: {json.dumps(decision)}")
            continue

        for method in ("setwise", "listwise", "bubble"):
            completed[f"{method}:{active_budget}"] = run_condition(
                dataset=dataset, method=method, budget=active_budget, seed=args.seed,
                queries=queries, documents=documents, engine=engine,
                base_signature=signature, resume=args.resume,
            )

        qualifier = qualifying_method(completed, budget=active_budget)
        if qualifier is not None or args.no_early_stop:
            for budget in args.budgets[1:]:
                for method in METHOD_ORDER[1:]:
                    completed[f"{method}:{budget}"] = run_condition(
                        dataset=dataset, method=method, budget=budget, seed=args.seed,
                        queries=queries, documents=documents, engine=engine,
                        base_signature=signature, resume=args.resume,
                    )
        decision = {
            "dataset": dataset,
            "decision": "validate_on_fresh_queries" if qualifier else "continue_dataset_queue",
            "qualifying_method": qualifier,
            "active_budget": active_budget,
            "selected_query_ids": manifest["selected_query_ids"],
        }
        decisions.append(decision)
        print(f"DATASET DECISION: {json.dumps(decision)}")
        if qualifier and not args.no_early_stop:
            stop_globally = True
            break

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    decision_path = RESULTS_DIR / "screen_decisions.json"
    decision_path.write_text(json.dumps(decisions, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {decision_path}")
    if stop_globally:
        print("GLOBAL EARLY STOP: a dataset passed the pre-registered pilot gate")


if __name__ == "__main__":
    main()
