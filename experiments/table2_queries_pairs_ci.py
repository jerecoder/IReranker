#!/usr/bin/env python3
"""
Table 2: Paired (per-query) bootstrap "A beats B" with resamples over queries.

This script:
  1) Runs the required ranker/oracle conditions for each dataset + matrix model (+ seed).
  2) Computes per-query NDCG@k (via pytrec_eval) and checks it matches BEIR's mean.
  3) Seed-averages per-query NDCG for each condition.
  4) For each (A,B) comparison:
       - computes per-query diffs d(q) = s_A(q) - s_B(q)
       - paired bootstrap over queries to get delta, CI, p-value
  5) Also computes a macro (unweighted mean) delta across datasets, with bootstrap.

Key change requested:
  - The "Comparison" field is generated automatically from the chosen A/B ranker+oracle
    so changing A/B automatically changes the matrix entry name.

Outputs:
  OUT_DIR/
    - raw_runs.csv
    - query_ndcg.csv
    - paired_bootstrap_pairs.csv
    - paired_bootstrap_avg.csv
"""

from __future__ import annotations

import argparse
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from ireranker.data.loaders import load_beir_dataset
from ireranker.evaluation.beir import dataset_to_beir_qrels, ranker_results_to_beir
from ireranker.rankers import get_ranker
from ireranker.oracles import BidirectionalMatrixOracle, SamplingMatrixOracle

def _slug(s: str) -> str:
    """
    Stable, CSV/file friendly identifier from a free-form label.
    """
    s = s.lower().strip()
    s = s.replace("+", " plus ")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

DEFAULT_SPLIT = "test"
DEFAULT_K = 10
DEFAULT_MODELS = ["flan-t5-large", "flan-t5-xl"]

DATASETS = [
    "trec-covid",
    # "robust04",  # keep commented if your download is broken
    "webis-touche2020",
    "scifact",
    "dbpedia-entity",
    "dl-2019",
    "dl-2020",
]

# -------------------------
# Define comparisons here
# -------------------------
A_RANKER = "mohajer + bubble"
PRP_BASELINE = "heap sort (classic)"

# Each tuple is (A_ranker, A_oracle, B_ranker, B_oracle)
COMPARE_PAIRS: List[Tuple[str, str, str, str]] = [
    (A_RANKER, "Sampling", PRP_BASELINE, "Bidirectional"),
    (A_RANKER, "Sampling", A_RANKER, "Bidirectional"),
]

# Output directory (keep manual or make dynamic if you prefer)
OUT_DIR = Path("reports/significance_testing/paired_bootstrap") / f"table2_vs_{_slug(PRP_BASELINE)}"
RAW_PATH = OUT_DIR / "raw_runs.csv"
QUERY_PATH = OUT_DIR / "query_ndcg.csv"
SIG_PATH = OUT_DIR / "paired_bootstrap_pairs.csv"
AVG_PATH = OUT_DIR / "paired_bootstrap_avg.csv"


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



def comparison_name(a_ranker: str, a_oracle: str, b_ranker: str, b_oracle: str) -> str:
    """
    Automatically generated comparison label for matrix entries / CSV fields.
    """
    return f"{_slug(a_ranker)}_{_slug(a_oracle)}_vs_{_slug(b_ranker)}_{_slug(b_oracle)}"


def paired_bootstrap_delta(
    diffs: np.ndarray,
    *,
    resamples: int,
    alpha: float,
    rng: np.random.Generator,
) -> Dict[str, float]:
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    n = int(d.size)
    if n == 0:
        return {
            "delta": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "ci_half": np.nan,
            "p": np.nan,
            "n": 0,
        }
    if n == 1:
        delta = float(d[0])
        return {"delta": delta, "ci_low": delta, "ci_high": delta, "ci_half": 0.0, "p": 1.0, "n": 1}

    delta = float(d.mean())
    idx = rng.integers(0, n, size=(resamples, n))
    boot = d[idx].mean(axis=1)
    ci_low = float(np.quantile(boot, alpha / 2))
    ci_high = float(np.quantile(boot, 1 - alpha / 2))
    ci_half = 0.5 * (ci_high - ci_low)
    p = 2 * min(float(np.mean(boot <= 0.0)), float(np.mean(boot >= 0.0)))
    return {"delta": delta, "ci_low": ci_low, "ci_high": ci_high, "ci_half": ci_half, "p": p, "n": n}


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
    matrix_model: str
    dataset: str
    ranker: str
    oracle: str
    seed: int


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=DEFAULT_SPLIT)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--resamples", type=int, default=10_000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--bootstrap-seed", type=int, default=42)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    _ensure_dir(OUT_DIR)

    # Derive METHODS from COMPARE_PAIRS so you only change A/B in one place.
    method_set = set()
    for a_rk, a_oc, b_rk, b_oc in COMPARE_PAIRS:
        method_set.add((a_rk, a_oc))
        method_set.add((b_rk, b_oc))

    # Stable ordering
    METHODS = sorted(method_set, key=lambda x: (x[1], x[0]))

    raw_existing = pd.DataFrame() if args.no_resume else _read_csv(RAW_PATH)
    query_existing = pd.DataFrame() if args.no_resume else _read_csv(QUERY_PATH)

    raw_rows = raw_existing.to_dict("records") if not raw_existing.empty else []
    query_rows = query_existing.to_dict("records") if not query_existing.empty else []

    def key_raw(r: dict) -> RunKey:
        return RunKey(
            matrix_model=str(r["MatrixModel"]),
            dataset=str(r["Dataset"]),
            ranker=str(r["Ranker"]),
            oracle=str(r["Oracle"]),
            seed=int(r["Seed"]),
        )

    existing_raw = {key_raw(r) for r in raw_rows} if raw_rows else set()

    def key_query(r: dict) -> RunKey:
        return RunKey(
            matrix_model=str(r["MatrixModel"]),
            dataset=str(r["Dataset"]),
            ranker=str(r["Ranker"]),
            oracle=str(r["Oracle"]),
            seed=int(r["Seed"]),
        )

    existing_query = {key_query(r) for r in query_rows} if query_rows else set()

    def have_run(mm: str, ds: str, rk: str, oc: str, s: int) -> bool:
        k = RunKey(mm, ds, rk, oc, int(s))
        return (k in existing_raw) and (k in existing_query)

    total = 0
    for mm in args.models:
        for ds in DATASETS:
            for rk, oc in METHODS:
                for s in args.seeds:
                    if not have_run(mm, ds, rk, oc, s):
                        total += 1

    pbar = tqdm(total=total, desc="Table 2 runs", unit="run")

    for mm in args.models:
        for ds in DATASETS:
            try:
                dataset = load_beir_dataset(ds, split=args.split, matrix_model=mm)
            except Exception as e:
                print(f"Skipping {mm}/{ds} due to load failure: {e}")
                continue

            task_qids = [t.query_id for t in dataset.tasks]

            for s in args.seeds:
                for rk, oc in METHODS:
                    if have_run(mm, ds, rk, oc, s):
                        continue

                    pbar.set_description(f"{mm}/{ds} | {rk} | {oc} | S={s}")

                    if oc == "Bidirectional":
                        oracle = BidirectionalMatrixOracle()
                    elif oc == "Sampling":
                        oracle = SamplingMatrixOracle(seed=s)
                    else:
                        raise ValueError(f"Unknown oracle type: {oc!r}")

                    ranker = get_ranker(rk, oracle=oracle, seed=s)
                    ranker.set_dataset(ds, split=args.split, query_ids=task_qids, matrix_model=mm)

                    mean_ndcg, comps, avg_comp, perq = evaluate_ranker_with_per_query_ndcg(
                        ranker=ranker, dataset=dataset, seed=s, k=args.k
                    )

                    raw_rows.append(
                        {
                            "MatrixModel": mm,
                            "Dataset": ds,
                            "Ranker": rk,
                            "Oracle": oc,
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
                                "MatrixModel": mm,
                                "Dataset": ds,
                                "Ranker": rk,
                                "Oracle": oc,
                                "Seed": int(s),
                                "QueryID": str(qid),
                                f"NDCG@{args.k}_query": float(v),
                            }
                        )
                    existing_query.add(RunKey(mm, ds, rk, oc, int(s)))
                    pbar.update(1)

    pbar.close()

    raw_df = pd.DataFrame(raw_rows).sort_values(
        ["MatrixModel", "Oracle", "Ranker", "Dataset", "Seed"]
    ).reset_index(drop=True)

    query_df = pd.DataFrame(query_rows).sort_values(
        ["MatrixModel", "Oracle", "Ranker", "Dataset", "Seed", "QueryID"]
    ).reset_index(drop=True)

    raw_df.to_csv(RAW_PATH, index=False)
    query_df.to_csv(QUERY_PATH, index=False)

    # Seed-averaged per-query series for each condition
    cond = {}
    score_col = f"NDCG@{args.k}_query"
    for (mm, ds, oc, rk), g in query_df.groupby(["MatrixModel", "Dataset", "Oracle", "Ranker"], sort=True):
        pivot = g.pivot_table(index="QueryID", columns="Seed", values=score_col, aggfunc="first")
        cond[(mm, ds, oc, rk)] = pivot.mean(axis=1, skipna=True)

    rng = np.random.default_rng(args.bootstrap_seed)

    # Per-dataset paired bootstrap results
    sig_rows = []
    for mm in args.models:
        for ds in DATASETS:
            for a_rk, a_oc, b_rk, b_oc in COMPARE_PAIRS:
                A = cond.get((mm, ds, a_oc, a_rk))
                B = cond.get((mm, ds, b_oc, b_rk))
                if A is None or B is None or A.empty or B.empty:
                    continue

                a, b = A.align(B, join="inner")
                diffs = (a - b).to_numpy(float)

                stats = paired_bootstrap_delta(diffs, resamples=args.resamples, alpha=args.alpha, rng=rng)
                sig_rows.append(
                    {
                        "MatrixModel": mm,
                        "Dataset": ds,
                        "Comparison": comparison_name(a_rk, a_oc, b_rk, b_oc),
                        "A_ranker": a_rk,
                        "A_oracle": a_oc,
                        "B_ranker": b_rk,
                        "B_oracle": b_oc,
                        "delta": stats["delta"],
                        "ci_low": stats["ci_low"],
                        "ci_high": stats["ci_high"],
                        "ci_half": stats["ci_half"],
                        "p": stats["p"],
                        "n_queries": stats["n"],
                        "sig_p05": bool(np.isfinite(stats["p"]) and stats["p"] < 0.05),
                    }
                )

    sig_df = pd.DataFrame(sig_rows)
    if not sig_df.empty:
        for c in ["delta", "ci_low", "ci_high", "ci_half"]:
            sig_df[c + "_pct"] = 100.0 * sig_df[c]
    sig_df.to_csv(SIG_PATH, index=False)

    # Macro across datasets (unweighted mean of dataset deltas)
    avg_rows = []
    for mm in args.models:
        for a_rk, a_oc, b_rk, b_oc in COMPARE_PAIRS:
            diffs_by_ds = []
            used = []

            for ds in DATASETS:
                A = cond.get((mm, ds, a_oc, a_rk))
                B = cond.get((mm, ds, b_oc, b_rk))
                if A is None or B is None or A.empty or B.empty:
                    continue

                a, b = A.align(B, join="inner")
                d = (a - b).to_numpy(float)
                d = d[np.isfinite(d)]
                if d.size == 0:
                    continue

                diffs_by_ds.append(d)
                used.append(ds)

            if not diffs_by_ds:
                continue

            obs = float(np.mean([d.mean() for d in diffs_by_ds]))

            boot = np.empty(args.resamples, dtype=float)
            for i in range(args.resamples):
                ds_means = []
                for d in diffs_by_ds:
                    n = d.size
                    idx = rng.integers(0, n, size=n)
                    ds_means.append(float(d[idx].mean()))
                boot[i] = float(np.mean(ds_means))

            ci_low = float(np.quantile(boot, args.alpha / 2))
            ci_high = float(np.quantile(boot, 1 - args.alpha / 2))
            ci_half = 0.5 * (ci_high - ci_low)
            p = 2 * min(float(np.mean(boot <= 0.0)), float(np.mean(boot >= 0.0)))

            avg_rows.append(
                {
                    "MatrixModel": mm,
                    "Comparison": comparison_name(a_rk, a_oc, b_rk, b_oc),
                    "A_ranker": a_rk,
                    "A_oracle": a_oc,
                    "B_ranker": b_rk,
                    "B_oracle": b_oc,
                    "Datasets_used": ",".join(used),
                    "n_datasets": int(len(used)),
                    "delta": obs,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "ci_half": ci_half,
                    "p": p,
                    "sig_p05": bool(np.isfinite(p) and p < 0.05),
                }
            )

    avg_df = pd.DataFrame(avg_rows)
    if not avg_df.empty:
        for c in ["delta", "ci_low", "ci_high", "ci_half"]:
            avg_df[c + "_pct"] = 100.0 * avg_df[c]
    avg_df.to_csv(AVG_PATH, index=False)

    print(f"Saved: {RAW_PATH}")
    print(f"Saved: {QUERY_PATH}")
    print(f"Saved: {SIG_PATH}")
    print(f"Saved: {AVG_PATH}")


if __name__ == "__main__":
    main()
