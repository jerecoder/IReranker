#!/usr/bin/env python3
"""
Table 1: Paired (per-query) bootstrap "A beats B" with resamples over queries.

ONLY comparison produced:
  - (A) mohajer + bubble  vs  (B) heap sort (classic)
    computed within each oracle block (Bidirectional and Sampling), for each budget.
Macro aggregation matches Table 1: bootstrap DL19 and DL20 queries separately then average.

Key change requested:
  - The "Comparison" field is generated automatically from the chosen A/B rankers
    (and oracle, if you want it included). So changing A_RANKER/B_RANKER changes
    the matrix entry name automatically.

Outputs:
  reports/significance_testing/paired_bootstrap/table1/
    - raw_runs.csv
    - query_ndcg.csv
    - paired_bootstrap_pairs.csv
"""

from __future__ import annotations

import argparse
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from ireranker.data.loaders import load_beir_dataset
from ireranker.evaluation.beir import dataset_to_beir_qrels, ranker_results_to_beir
from ireranker.rankers import get_ranker
from ireranker.oracles import BidirectionalMatrixOracle, SamplingMatrixOracle


DEFAULT_SPLIT = "test"
DEFAULT_K = 10
DEFAULT_MODEL = "flan-t5-xl"

DATASETS = ["dl-2019", "dl-2020"]
BUDGETS = [100, 150, 200, 250, 300, 350, 400, 450, 500]

A_RANKER = "mohajer + bubble"
B_RANKER = "heap sort (classic)"

# We still need to run these rankers to compute the comparison:
RANKERS_TO_RUN = [A_RANKER, B_RANKER]

OUT_DIR = Path("reports/significance_testing/paired_bootstrap/table1")
RAW_PATH = OUT_DIR / "raw_runs.csv"
QUERY_PATH = OUT_DIR / "query_ndcg.csv"
SIG_PATH = OUT_DIR / "paired_bootstrap_pairs.csv"


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def _require_pytrec_eval():
    try:
        import pytrec_eval  # type: ignore
    except Exception as e:
        raise RuntimeError(f"pytrec_eval is required. Install it. Error: {e!r}")
    return pytrec_eval


def _slug(s: str) -> str:
    """
    Stable, CSV-friendly identifier from a free-form label.
    """
    s = s.lower().strip()
    s = s.replace("+", " plus ")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def comparison_name(a_ranker: str, b_ranker: str) -> str:
    """
    Automatically generated comparison label (independent of oracle/budget).
    Oracle and budget are already separate columns in the output.
    """
    return f"{_slug(a_ranker)}_vs_{_slug(b_ranker)}"


def paired_bootstrap_delta_macro_dl19_dl20(
    diffs_dl19: np.ndarray,
    diffs_dl20: np.ndarray,
    *,
    resamples: int,
    alpha: float,
    rng: np.random.Generator,
) -> Dict[str, float]:
    d19 = np.asarray(diffs_dl19, dtype=float)
    d20 = np.asarray(diffs_dl20, dtype=float)
    d19 = d19[np.isfinite(d19)]
    d20 = d20[np.isfinite(d20)]
    n19, n20 = int(d19.size), int(d20.size)
    if n19 == 0 or n20 == 0:
        return {"delta": np.nan, "ci_low": np.nan, "ci_high": np.nan, "ci_half": np.nan, "p": np.nan, "n": 0}

    delta = 0.5 * (float(d19.mean()) + float(d20.mean()))
    idx19 = rng.integers(0, n19, size=(resamples, n19))
    idx20 = rng.integers(0, n20, size=(resamples, n20))
    boot = 0.5 * (d19[idx19].mean(axis=1) + d20[idx20].mean(axis=1))

    ci_low = float(np.quantile(boot, alpha / 2))
    ci_high = float(np.quantile(boot, 1 - alpha / 2))
    ci_half = 0.5 * (ci_high - ci_low)
    p = 2 * min(float(np.mean(boot <= 0.0)), float(np.mean(boot >= 0.0)))

    return {
        "delta": float(delta),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_half": ci_half,
        "p": p,
        "n": int(n19 + n20),
    }


def paired_bootstrap_delta_macro(
    diffs_by_dataset: Mapping[str, np.ndarray],
    *,
    resamples: int,
    alpha: float,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Bootstrap queries within each dataset, then macro-average datasets.

    With one dataset this is the ordinary paired per-query bootstrap. With
    multiple datasets, each dataset contributes equal weight regardless of its
    query count, matching the original DL19/DL20 calculation.
    """
    finite_diffs = []
    for diffs in diffs_by_dataset.values():
        values = np.asarray(diffs, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return {
                "delta": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "ci_half": np.nan,
                "p": np.nan,
                "n": 0,
            }
        finite_diffs.append(values)

    if not finite_diffs:
        return {
            "delta": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "ci_half": np.nan,
            "p": np.nan,
            "n": 0,
        }

    delta = float(np.mean([values.mean() for values in finite_diffs]))
    boot_means = []
    for values in finite_diffs:
        indices = rng.integers(0, values.size, size=(resamples, values.size))
        boot_means.append(values[indices].mean(axis=1))
    boot = np.mean(np.stack(boot_means, axis=0), axis=0)

    ci_low = float(np.quantile(boot, alpha / 2))
    ci_high = float(np.quantile(boot, 1 - alpha / 2))
    return {
        "delta": delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_half": 0.5 * (ci_high - ci_low),
        "p": 2 * min(float(np.mean(boot <= 0.0)), float(np.mean(boot >= 0.0))),
        "n": int(sum(values.size for values in finite_diffs)),
    }


def evaluate_ranker_with_per_query_ndcg(
    *,
    ranker,
    dataset,
    seed: int,
    k: int,
    mean_tol: float = 1e-4,
) -> Tuple[float, int, float, Dict[str, float]]:
    from beir.retrieval.evaluation import EvaluateRetrieval  # type: ignore

    pytrec_eval = _require_pytrec_eval()
    qrels = dataset_to_beir_qrels(dataset)
    task_count = len(dataset.tasks)

    ranker.set_seed(seed)
    ranker.reset_comparisons()
    run = ranker_results_to_beir(ranker, dataset, random.Random(seed))

    ndcg, *_ = EvaluateRetrieval.evaluate(qrels, run, [k])
    mean_ndcg = float(ndcg.get(f"NDCG@{k}", 0.0))

    comps = int(ranker.comparisons)
    avg_comp = float(comps / task_count) if task_count else 0.0

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {f"ndcg_cut.{k}"})
    perq = evaluator.evaluate(run)
    key = f"ndcg_cut_{k}"
    per_query = {qid: float(perq.get(qid, {}).get(key, 0.0)) for qid in qrels.keys()}

    if per_query:
        check_mean = float(np.mean(list(per_query.values())))
        if abs(check_mean - mean_ndcg) > mean_tol:
            raise RuntimeError(f"Mean mismatch: BEIR={mean_ndcg:.8f} vs per-query={check_mean:.8f}")

    return mean_ndcg, comps, avg_comp, per_query


@dataclass(frozen=True)
class RunKey:
    dataset: str
    ranker: str
    oracle: str
    budget: int
    seed: int
    matrix_model: str


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=DEFAULT_SPLIT)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--resamples", type=int, default=10_000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--bootstrap-seed", type=int, default=42)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    _ensure_dir(OUT_DIR)

    raw_existing = pd.DataFrame() if args.no_resume else _read_csv(RAW_PATH)
    query_existing = pd.DataFrame() if args.no_resume else _read_csv(QUERY_PATH)

    raw_rows = raw_existing.to_dict("records") if not raw_existing.empty else []
    query_rows = query_existing.to_dict("records") if not query_existing.empty else []

    def key_raw(r: dict) -> RunKey:
        return RunKey(
            dataset=str(r["Dataset"]),
            ranker=str(r["Ranker"]),
            oracle=str(r["Oracle"]),
            budget=int(r["Budget"]),
            seed=int(r["Seed"]),
            matrix_model=str(r.get("MatrixModel", args.model)),
        )

    existing_raw = {key_raw(r) for r in raw_rows} if raw_rows else set()

    def key_query(r: dict) -> RunKey:
        return RunKey(
            dataset=str(r["Dataset"]),
            ranker=str(r["Ranker"]),
            oracle=str(r["Oracle"]),
            budget=int(r["Budget"]),
            seed=int(r["Seed"]),
            matrix_model=str(r.get("MatrixModel", args.model)),
        )

    existing_query = {key_query(r) for r in query_rows} if query_rows else set()

    def have_run(ds: str, rk: str, oc: str, b: int, s: int) -> bool:
        k = RunKey(ds, rk, oc, int(b), int(s), args.model)
        return (k in existing_raw) and (k in existing_query)

    total = 0
    for ds in DATASETS:
        for rk in RANKERS_TO_RUN:
            for b in BUDGETS:
                for s in args.seeds:
                    for oc in ("Bidirectional", "Sampling"):
                        if not have_run(ds, rk, oc, b, s):
                            total += 1

    pbar = tqdm(total=total, desc="Table 1 runs", unit="run")

    for ds in DATASETS:
        dataset = load_beir_dataset(ds, split=args.split, matrix_model=args.model)
        task_qids = [t.query_id for t in dataset.tasks]

        for s in args.seeds:
            for rk in RANKERS_TO_RUN:
                for b in BUDGETS:
                    # Bidirectional
                    if not have_run(ds, rk, "Bidirectional", b, s):
                        pbar.set_description(f"{ds} | {rk} | Bidirectional | B={b} | S={s}")
                        oracle = BidirectionalMatrixOracle(comparison_limit=b, comparison_limit_per_task=True)
                        ranker = get_ranker(rk, oracle=oracle, seed=s)
                        ranker.set_dataset(ds, split=args.split, query_ids=task_qids, matrix_model=args.model)

                        mean_ndcg, comps, avg_comp, perq = evaluate_ranker_with_per_query_ndcg(
                            ranker=ranker, dataset=dataset, seed=s, k=args.k
                        )

                        raw_rows.append(
                            {
                                "MatrixModel": args.model,
                                "Dataset": ds,
                                "Ranker": rk,
                                "Oracle": "Bidirectional",
                                "Budget": int(b),
                                "Seed": int(s),
                                f"NDCG@{args.k}": float(mean_ndcg),
                                "Comparisons": int(comps),
                                "average_comparison_per_task": float(avg_comp),
                            }
                        )
                        existing_raw.add(key_raw(raw_rows[-1]))

                        for qid, v in perq.items():
                            query_rows.append(
                                {
                                    "MatrixModel": args.model,
                                    "Dataset": ds,
                                    "Ranker": rk,
                                    "Oracle": "Bidirectional",
                                    "Budget": int(b),
                                    "Seed": int(s),
                                    "QueryID": str(qid),
                                    f"NDCG@{args.k}_query": float(v),
                                }
                            )
                        existing_query.add(RunKey(ds, rk, "Bidirectional", int(b), int(s), args.model))
                        pbar.update(1)

                    # Sampling
                    if not have_run(ds, rk, "Sampling", b, s):
                        pbar.set_description(f"{ds} | {rk} | Sampling | B={b} | S={s}")
                        oracle = SamplingMatrixOracle(seed=s, comparison_limit=b, comparison_limit_per_task=True)
                        ranker = get_ranker(rk, oracle=oracle, seed=s)
                        ranker.set_dataset(ds, split=args.split, query_ids=task_qids, matrix_model=args.model)

                        mean_ndcg, comps, avg_comp, perq = evaluate_ranker_with_per_query_ndcg(
                            ranker=ranker, dataset=dataset, seed=s, k=args.k
                        )

                        raw_rows.append(
                            {
                                "MatrixModel": args.model,
                                "Dataset": ds,
                                "Ranker": rk,
                                "Oracle": "Sampling",
                                "Budget": int(b),
                                "Seed": int(s),
                                f"NDCG@{args.k}": float(mean_ndcg),
                                "Comparisons": int(comps),
                                "average_comparison_per_task": float(avg_comp),
                            }
                        )
                        existing_raw.add(key_raw(raw_rows[-1]))

                        for qid, v in perq.items():
                            query_rows.append(
                                {
                                    "MatrixModel": args.model,
                                    "Dataset": ds,
                                    "Ranker": rk,
                                    "Oracle": "Sampling",
                                    "Budget": int(b),
                                    "Seed": int(s),
                                    "QueryID": str(qid),
                                    f"NDCG@{args.k}_query": float(v),
                                }
                            )
                        existing_query.add(RunKey(ds, rk, "Sampling", int(b), int(s), args.model))
                        pbar.update(1)

    pbar.close()

    raw_df = pd.DataFrame(raw_rows).sort_values(
        ["MatrixModel", "Oracle", "Ranker", "Dataset", "Budget", "Seed"]
    ).reset_index(drop=True)

    query_df = pd.DataFrame(query_rows).sort_values(
        ["MatrixModel", "Oracle", "Ranker", "Dataset", "Budget", "Seed", "QueryID"]
    ).reset_index(drop=True)

    raw_df.to_csv(RAW_PATH, index=False)
    query_df.to_csv(QUERY_PATH, index=False)

    # Build seed-averaged per-query series for each condition
    cond = {}
    score_col = f"NDCG@{args.k}_query"
    for (ds, oc, rk, b), g in query_df.groupby(["Dataset", "Oracle", "Ranker", "Budget"], sort=True):
        pivot = g.pivot_table(index="QueryID", columns="Seed", values=score_col, aggfunc="first")
        cond[(ds, oc, rk, int(b))] = pivot.mean(axis=1, skipna=True)

    rng = np.random.default_rng(args.bootstrap_seed)
    sig_rows = []
    comp_label = comparison_name(A_RANKER, B_RANKER)

    for oc in ["Bidirectional", "Sampling"]:
        for b in BUDGETS:
            diffs_by_dataset = {}
            for ds in DATASETS:
                a_scores = cond.get((ds, oc, A_RANKER, int(b)))
                b_scores = cond.get((ds, oc, B_RANKER, int(b)))
                if a_scores is None or b_scores is None:
                    break
                aligned_a, aligned_b = a_scores.align(b_scores, join="inner")
                diffs_by_dataset[ds] = (aligned_a - aligned_b).to_numpy(float)
            if len(diffs_by_dataset) != len(DATASETS):
                continue

            stats = paired_bootstrap_delta_macro(
                diffs_by_dataset,
                resamples=args.resamples,
                alpha=args.alpha,
                rng=rng,
            )

            sig_rows.append(
                {
                    "Comparison": comp_label,
                    "Oracle": oc,
                    "Budget": int(b),
                    "A": A_RANKER,
                    "B": B_RANKER,
                    "delta": stats["delta"],
                    "ci_low": stats["ci_low"],
                    "ci_high": stats["ci_high"],
                    "ci_half": stats["ci_half"],
                    "p": stats["p"],
                    "n_queries_total": stats["n"],
                    "sig_p05": bool(np.isfinite(stats["p"]) and stats["p"] < 0.05),
                }
            )

    sig_df = pd.DataFrame(sig_rows)
    if not sig_df.empty:
        for c in ["delta", "ci_low", "ci_high", "ci_half"]:
            sig_df[c + "_pct"] = 100.0 * sig_df[c]
    sig_df.to_csv(SIG_PATH, index=False)

    print(f"Saved: {RAW_PATH}")
    print(f"Saved: {QUERY_PATH}")
    print(f"Saved: {SIG_PATH}")


if __name__ == "__main__":
    main()
