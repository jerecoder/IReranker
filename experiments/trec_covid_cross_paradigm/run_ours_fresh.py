#!/usr/bin/env python3
"""Run fresh-cache Mohajer points with per-run token and compute accounting."""

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.table1_queries_pairs_ci import evaluate_ranker_with_per_query_ndcg  # noqa: E402
from ireranker.data.loaders import load_beir_dataset_from_bm25
from ireranker.oracles import LiveFlanT5SamplingOracle
from ireranker.rankers import get_ranker


OUT = ROOT / "experiments/trec_covid_cross_paradigm/metrics/ours.csv"
CACHE_DIR = ROOT / "experiments/trec_covid_cross_paradigm/fresh-cache"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budgets", type=int, nargs="+", default=[100, 150, 200, 250, 300])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    dataset = load_beir_dataset_from_bm25("trec-covid", top_k=100)
    qids = [task.query_id for task in dataset.tasks]
    rows = []
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for budget in args.budgets:
        cache = CACHE_DIR / f"mohajer_sampling_b{budget}.pkl"
        cache.unlink(missing_ok=True)
        oracle = LiveFlanT5SamplingOracle(
            model_name="google/flan-t5-large",
            cache_path=cache,
            device=args.device,
            comparison_limit=budget,
            comparison_limit_per_task=True,
            cache_flush_interval=20,
        )
        ranker = get_ranker("mohajer (ir)", oracle=oracle, seed=args.seed)
        ranker.set_dataset("trec-covid", split="test", query_ids=qids, matrix_model="flan-t5-large")
        ndcg, comparisons, avg_comparisons, _ = evaluate_ranker_with_per_query_ndcg(
            ranker=ranker, dataset=dataset, seed=args.seed, k=10
        )
        rows.append({
            "method": "mohajer",
            "variant": "sampling",
            "budget": budget,
            "ndcg10": ndcg,
            "avg_calls": oracle.model_inferences / len(qids),
            "avg_comparisons": avg_comparisons,
            "avg_prompt_tokens": oracle.total_prompt_tokens / len(qids),
            "avg_completion_tokens": oracle.total_decoder_tokens / len(qids),
            "avg_total_tokens": (oracle.total_prompt_tokens + oracle.total_decoder_tokens) / len(qids),
            "avg_time_sec": oracle.total_inference_seconds / len(qids),
            "total_comparisons": comparisons,
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
