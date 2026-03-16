#!/usr/bin/env python3
"""
Paired (per-query) bootstrap significance tests: "A beats B"

What this script does
---------------------
1) Runs the SAME evaluation pipeline you use in the paper (ranker_results_to_beir + BEIR EvaluateRetrieval)
   and additionally logs per-query NDCG@10 for each run.
2) Computes paired-bootstrap confidence intervals over QUERIES (10,000 resamples) for mean deltas:
      delta = mean_q( NDCG_A(q) - NDCG_B(q) )
   - For Table 1 (DL19+DL20 macro): bootstrap queries within each dataset separately, then macro-average.
   - For Table 2: bootstrap within each dataset separately (and optionally macro across datasets for "Avg").

Ranker pairs (as discussed, with PRP baseline = heap sort classic):
------------------------------------------------------------------
Table 1 (per budget, per oracle block):
  - Mohajer(ir) vs BubbleSort(classic)     (within each oracle)
  - Mohajer(ir) vs HeapSort(classic)       (within each oracle)  <-- PRP baseline
  - Mohajer(ir) Sampling vs Mohajer(ir) Bidirectional           <-- oracle effect

Table 2 (per dataset, per model):
  - Mohajer(ir) Sampling vs BubbleSort(classic) Bidirectional   <-- baseline
  - Mohajer(ir) Sampling vs Mohajer(ir) Bidirectional           <-- oracle effect

Notes
-----
- Seeds: if you provide multiple seeds, the script averages per-query NDCG across seeds first,
  then runs the paired bootstrap over queries.
- If you want EXACT paper means, set SEEDS=[42] and use the same matrices/caches as paper runs.

Outputs
-------
reports/significance_testing/paired_bootstrap/table1/
  - raw_runs.csv
  - query_ndcg.csv
  - paired_bootstrap_pairs.csv

reports/significance_testing/paired_bootstrap/table2/
  - raw_runs.csv
  - query_ndcg.csv
  - paired_bootstrap_pairs.csv
  - paired_bootstrap_avg.csv  (optional macro across datasets for the Avg column)

Usage examples
--------------
# Table 1 only (budgeted DL19/DL20)
python paired_bootstrap_significance.py --table table1

# Table 2 only (end-to-end across datasets)
python paired_bootstrap_significance.py --table table2

# Both
python paired_bootstrap_significance.py --table both

# Use multiple seeds (seed-averaged per-query scores, then bootstrap over queries)
python paired_bootstrap_significance.py --table table1 --seeds 40 41 42 43 44 45 46 47 48 49
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from ireranker.data.loaders import load_beir_dataset
from ireranker.evaluation.beir import dataset_to_beir_qrels, ranker_results_to_beir
from ireranker.rankers import get_ranker
from ireranker.oracles import BidirectionalMatrixOracle, SamplingMatrixOracle


# =========================
# Defaults / paper alignment
# =========================
DEFAULT_SPLIT = "test"
DEFAULT_K = 10
DEFAULT_MODEL_TABLE1 = "flan-t5-xl"  # Table 1 in your paper excerpt
DEFAULT_MODELS_TABLE2 = ["flan-t5-large", "flan-t5-xl"]

DEFAULT_DATASETS_TABLE1 = ["dl-2019", "dl-2020"]
DEFAULT_BUDGETS_TABLE1 = [100, 150, 200, 250, 300, 350, 400, 450, 500]
DEFAULT_RANKERS_TABLE1 = [
    "mohajer (ir)",
    "mohajer + bubble",
    "pac + bubble",
    "bubble sort (classic)",
    "quick sort (classic)",
    "heap sort (classic)",  # PRP baseline name you requested
]

# Table 2 datasets (from your paper table)
DEFAULT_DATASETS_TABLE2 = [
    "trec-covid",
    # "robust04",
    "webis-touche2020",
    "scifact",
    "dbpedia-entity",
    "dl-2019",
    "dl-2020",
]

# Main table2 methods we need to run to support the pair tests
# (We only run what's needed for the comparisons defined below.)
TABLE2_METHODS = [
    # Baseline
    ("bubble sort (classic)", "Bidirectional"),
    # Mohajer variants needed
    ("mohajer (ir)", "Bidirectional"),
    ("mohajer (ir)", "Sampling"),
]


# =========================
# Paired bootstrap helpers
# =========================
def paired_bootstrap_delta(
    diffs: np.ndarray,
    *,
    resamples: int,
    alpha: float,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Bootstrap CI for mean(diffs) by resampling queries with replacement."""
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    n = int(d.size)
    if n == 0:
        return {"delta": np.nan, "ci_low": np.nan, "ci_high": np.nan, "ci_half": np.nan, "p": np.nan, "n": 0}
    if n == 1:
        delta = float(d[0])
        return {"delta": delta, "ci_low": delta, "ci_high": delta, "ci_half": 0.0, "p": 1.0, "n": 1}

    delta = float(d.mean())
    idx = rng.integers(0, n, size=(resamples, n))
    boot = d[idx].mean(axis=1)

    ci_low = float(np.quantile(boot, alpha / 2))
    ci_high = float(np.quantile(boot, 1 - alpha / 2))
    ci_half = 0.5 * (ci_high - ci_low)

    # Two-sided bootstrap p-value for H0: delta = 0
    p = 2 * min(float(np.mean(boot <= 0.0)), float(np.mean(boot >= 0.0)))

    return {"delta": delta, "ci_low": ci_low, "ci_high": ci_high, "ci_half": ci_half, "p": p, "n": n}


def paired_bootstrap_delta_macro_dl19_dl20(
    diffs_dl19: np.ndarray,
    diffs_dl20: np.ndarray,
    *,
    resamples: int,
    alpha: float,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """
    Macro delta consistent with your Table 1 macro aggregation:
      delta_macro = 0.5*(mean_dl19 + mean_dl20)
    Bootstrap queries within each dataset separately, then macro-average.
    """
    d19 = np.asarray(diffs_dl19, dtype=float)
    d20 = np.asarray(diffs_dl20, dtype=float)
    d19 = d19[np.isfinite(d19)]
    d20 = d20[np.isfinite(d20)]

    n19 = int(d19.size)
    n20 = int(d20.size)
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

    return {"delta": float(delta), "ci_low": ci_low, "ci_high": ci_high, "ci_half": ci_half, "p": p, "n": int(n19 + n20)}


# =========================
# Evaluation with per-query logging (matches BEIR evaluation)
# =========================
def _require_pytrec_eval():
    try:
        import pytrec_eval  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "pytrec_eval is required for per-query NDCG. Install it in your env. "
            f"Original error: {e!r}"
        )
    return pytrec_eval


def evaluate_ranker_with_per_query_ndcg(
    *,
    ranker,
    dataset,
    seed: int,
    k: int,
) -> Tuple[float, int, float, Dict[str, float]]:
    """
    Runs ranking once and returns:
      - mean_ndcg@k (BEIR style, macro over queries in qrels)
      - total comparisons (as counted by the ranker/oracle)
      - avg comparisons per task
      - per-query ndcg@k dict for all qids in qrels
    """
    # BEIR evaluator used in your repo (same dependency your paper pipeline relies on)
    from beir.retrieval.evaluation import EvaluateRetrieval  # type: ignore

    pytrec_eval = _require_pytrec_eval()

    qrels = dataset_to_beir_qrels(dataset)
    task_count = len(dataset.tasks)

    ranker.set_seed(seed)
    ranker.reset_comparisons()

    res = ranker_results_to_beir(ranker, dataset, random.Random(seed))

    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, res, [k])
    mean_ndcg = float(ndcg.get(f"NDCG@{k}", 0.0))

    total_comparisons = int(ranker.comparisons)
    avg_comparisons = float(total_comparisons / task_count) if task_count else 0.0

    # Per-query ndcg_cut.k using the same qrels + run.
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {f"ndcg_cut.{k}"})
    perq = evaluator.evaluate(res)
    key = f"ndcg_cut_{k}"

    # Ensure we output exactly the same query set as the BEIR mean (qrels keys)
    per_query_ndcg: Dict[str, float] = {}
    for qid in qrels.keys():
        per_query_ndcg[qid] = float(perq.get(qid, {}).get(key, 0.0))

    # Sanity check: mean(per-query) should match BEIR mean (within tiny tolerance)
    # If this fails, you likely have an environment mismatch; do not proceed silently.
    if per_query_ndcg:
        check_mean = float(np.mean(list(per_query_ndcg.values())))
        if abs(check_mean - mean_ndcg) > 1e-4:
            raise RuntimeError(
                f"Mean mismatch: BEIR mean={mean_ndcg:.8f} vs mean(per-query)={check_mean:.8f}. "
                "This indicates evaluation inconsistency; fix before using significance results."
            )

    return mean_ndcg, total_comparisons, avg_comparisons, per_query_ndcg


# =========================
# Storage / resume
# =========================
@dataclass(frozen=True)
class T1Key:
    dataset: str
    ranker: str
    oracle: str
    budget: int
    seed: int
    matrix_model: str


@dataclass(frozen=True)
class T2Key:
    dataset: str
    ranker: str
    oracle: str
    seed: int
    matrix_model: str


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# =========================
# Table 1 runner (budget sweep) + pair tests
# =========================
def run_table1_and_pairs(
    *,
    out_dir: Path,
    datasets: List[str],
    budgets: List[int],
    rankers: List[str],
    seeds: List[int],
    matrix_model: str,
    split: str,
    k: int,
    resamples: int,
    alpha: float,
    bootstrap_seed: int,
    no_resume: bool,
) -> None:
    _ensure_dir(out_dir)
    raw_path = out_dir / "raw_runs.csv"
    query_path = out_dir / "query_ndcg.csv"
    sig_path = out_dir / "paired_bootstrap_pairs.csv"

    raw_existing = pd.DataFrame() if no_resume else _read_csv(raw_path)
    query_existing = pd.DataFrame() if no_resume else _read_csv(query_path)

    raw_rows = raw_existing.to_dict("records") if not raw_existing.empty else []
    query_rows = query_existing.to_dict("records") if not query_existing.empty else []

    def key_from_raw(r: dict) -> T1Key:
        return T1Key(
            dataset=str(r["Dataset"]),
            ranker=str(r["Ranker"]),
            oracle=str(r["Oracle"]),
            budget=int(r["Budget"]),
            seed=int(r["Seed"]),
            matrix_model=str(r.get("MatrixModel", matrix_model)),
        )

    existing_raw = {key_from_raw(r) for r in raw_rows} if raw_rows else set()

    def key_from_query(r: dict) -> T1Key:
        return T1Key(
            dataset=str(r["Dataset"]),
            ranker=str(r["Ranker"]),
            oracle=str(r["Oracle"]),
            budget=int(r["Budget"]),
            seed=int(r["Seed"]),
            matrix_model=str(r.get("MatrixModel", matrix_model)),
        )

    existing_query = {key_from_query(r) for r in query_rows} if query_rows else set()

    def have_run(ds: str, rk: str, oc: str, b: int, s: int) -> bool:
        k1 = T1Key(ds, rk, oc, int(b), int(s), matrix_model)
        return k1 in existing_raw and k1 in existing_query

    # Total remaining runs for progress
    total = 0
    for ds in datasets:
        for rk in rankers:
            for b in budgets:
                for s in seeds:
                    for oc in ("Bidirectional", "Sampling"):
                        if not have_run(ds, rk, oc, b, s):
                            total += 1

    pbar = tqdm(total=total, desc="Table 1: runs", unit="run")

    for ds_name in datasets:
        dataset = load_beir_dataset(ds_name, split=split, matrix_model=matrix_model)
        task_qids = [t.query_id for t in dataset.tasks]
        task_count = len(dataset.tasks)

        for s in seeds:
            for rk_name in rankers:
                for b in budgets:
                    # Bidirectional
                    if not have_run(ds_name, rk_name, "Bidirectional", b, s):
                        pbar.set_description(f"T1 {ds_name} | {rk_name} | Bidirectional | B={b} | S={s}")
                        oracle = BidirectionalMatrixOracle(comparison_limit=b, comparison_limit_per_task=True)
                        ranker = get_ranker(rk_name, oracle=oracle, seed=s)
                        ranker.set_dataset(ds_name, split=split, query_ids=task_qids, matrix_model=matrix_model)

                        mean_ndcg, comps, avg_comp, perq = evaluate_ranker_with_per_query_ndcg(
                            ranker=ranker, dataset=dataset, seed=s, k=k
                        )

                        raw_rows.append(
                            {
                                "MatrixModel": matrix_model,
                                "Dataset": ds_name,
                                "Ranker": rk_name,
                                "Oracle": "Bidirectional",
                                "Budget": int(b),
                                "Seed": int(s),
                                "NDCG@10": float(mean_ndcg),
                                "Comparisons": int(comps),
                                "average_comparison_per_task": float(avg_comp),
                            }
                        )
                        existing_raw.add(key_from_raw(raw_rows[-1]))

                        for qid, v in perq.items():
                            query_rows.append(
                                {
                                    "MatrixModel": matrix_model,
                                    "Dataset": ds_name,
                                    "Ranker": rk_name,
                                    "Oracle": "Bidirectional",
                                    "Budget": int(b),
                                    "Seed": int(s),
                                    "QueryID": str(qid),
                                    "NDCG@10_query": float(v),
                                }
                            )
                        existing_query.add(T1Key(ds_name, rk_name, "Bidirectional", int(b), int(s), matrix_model))
                        pbar.update(1)

                    # Sampling
                    if not have_run(ds_name, rk_name, "Sampling", b, s):
                        pbar.set_description(f"T1 {ds_name} | {rk_name} | Sampling | B={b} | S={s}")
                        oracle = SamplingMatrixOracle(seed=s, comparison_limit=b, comparison_limit_per_task=True)
                        ranker = get_ranker(rk_name, oracle=oracle, seed=s)
                        ranker.set_dataset(ds_name, split=split, query_ids=task_qids, matrix_model=matrix_model)

                        mean_ndcg, comps, avg_comp, perq = evaluate_ranker_with_per_query_ndcg(
                            ranker=ranker, dataset=dataset, seed=s, k=k
                        )

                        raw_rows.append(
                            {
                                "MatrixModel": matrix_model,
                                "Dataset": ds_name,
                                "Ranker": rk_name,
                                "Oracle": "Sampling",
                                "Budget": int(b),
                                "Seed": int(s),
                                "NDCG@10": float(mean_ndcg),
                                "Comparisons": int(comps),
                                "average_comparison_per_task": float(avg_comp),
                            }
                        )
                        existing_raw.add(key_from_raw(raw_rows[-1]))

                        for qid, v in perq.items():
                            query_rows.append(
                                {
                                    "MatrixModel": matrix_model,
                                    "Dataset": ds_name,
                                    "Ranker": rk_name,
                                    "Oracle": "Sampling",
                                    "Budget": int(b),
                                    "Seed": int(s),
                                    "QueryID": str(qid),
                                    "NDCG@10_query": float(v),
                                }
                            )
                        existing_query.add(T1Key(ds_name, rk_name, "Sampling", int(b), int(s), matrix_model))
                        pbar.update(1)

    pbar.close()

    raw_df = pd.DataFrame(raw_rows).sort_values(
        ["MatrixModel", "Oracle", "Ranker", "Dataset", "Budget", "Seed"]
    ).reset_index(drop=True)
    query_df = pd.DataFrame(query_rows).sort_values(
        ["MatrixModel", "Oracle", "Ranker", "Dataset", "Budget", "Seed", "QueryID"]
    ).reset_index(drop=True)

    raw_df.to_csv(raw_path, index=False)
    query_df.to_csv(query_path, index=False)

    # ------------------------------
    # Build per-query (seed-averaged) vectors for each condition
    # ------------------------------
    # condition key: (Dataset, Oracle, Ranker, Budget) -> Series indexed by QueryID
    cond = {}
    group_cols = ["Dataset", "Oracle", "Ranker", "Budget"]
    for (ds, oc, rk, b), g in query_df.groupby(group_cols, sort=True):
        pivot = g.pivot_table(index="QueryID", columns="Seed", values="NDCG@10_query", aggfunc="first")
        per_query_mean = pivot.mean(axis=1, skipna=True)
        cond[(ds, oc, rk, int(b))] = per_query_mean

    rng = np.random.default_rng(bootstrap_seed)
    sig_rows = []

    # Comparisons we want (Table 1 macro DL19+DL20)
    # (A oracle, A ranker) vs (B oracle, B ranker)
    for oc in ["Bidirectional", "Sampling"]:
        for b in budgets:
            # Mohajer vs BubbleSort
            A = cond.get(("dl-2019", oc, "mohajer (ir)", int(b)))
            B = cond.get(("dl-2019", oc, "bubble sort (classic)", int(b)))
            A2 = cond.get(("dl-2020", oc, "mohajer (ir)", int(b)))
            B2 = cond.get(("dl-2020", oc, "bubble sort (classic)", int(b)))
            if A is not None and B is not None and A2 is not None and B2 is not None:
                d19 = (A.align(B, join="inner")[0] - A.align(B, join="inner")[1]).to_numpy(float)
                d20 = (A2.align(B2, join="inner")[0] - A2.align(B2, join="inner")[1]).to_numpy(float)
                stats = paired_bootstrap_delta_macro_dl19_dl20(d19, d20, resamples=resamples, alpha=alpha, rng=rng)
                sig_rows.append(
                    {
                        "Table": "table1",
                        "Comparison": "Mohajer_vs_BubbleSort",
                        "Oracle": oc,
                        "Budget": int(b),
                        "A": "mohajer (ir)",
                        "B": "bubble sort (classic)",
                        "delta": stats["delta"],
                        "ci_low": stats["ci_low"],
                        "ci_high": stats["ci_high"],
                        "ci_half": stats["ci_half"],
                        "p": stats["p"],
                        "n_queries_total": stats["n"],
                        "sig_p05": bool(np.isfinite(stats["p"]) and stats["p"] < 0.05),
                    }
                )

            # Mohajer vs HeapSort (PRP baseline you requested)
            A = cond.get(("dl-2019", oc, "mohajer (ir)", int(b)))
            B = cond.get(("dl-2019", oc, "heap sort (classic)", int(b)))
            A2 = cond.get(("dl-2020", oc, "mohajer (ir)", int(b)))
            B2 = cond.get(("dl-2020", oc, "heap sort (classic)", int(b)))
            if A is not None and B is not None and A2 is not None and B2 is not None:
                d19 = (A.align(B, join="inner")[0] - A.align(B, join="inner")[1]).to_numpy(float)
                d20 = (A2.align(B2, join="inner")[0] - A2.align(B2, join="inner")[1]).to_numpy(float)
                stats = paired_bootstrap_delta_macro_dl19_dl20(d19, d20, resamples=resamples, alpha=alpha, rng=rng)
                sig_rows.append(
                    {
                        "Table": "table1",
                        "Comparison": "Mohajer_vs_HeapSort",
                        "Oracle": oc,
                        "Budget": int(b),
                        "A": "mohajer (ir)",
                        "B": "heap sort (classic)",
                        "delta": stats["delta"],
                        "ci_low": stats["ci_low"],
                        "ci_high": stats["ci_high"],
                        "ci_half": stats["ci_half"],
                        "p": stats["p"],
                        "n_queries_total": stats["n"],
                        "sig_p05": bool(np.isfinite(stats["p"]) and stats["p"] < 0.05),
                    }
                )

    # Oracle effect: Mohajer Sampling vs Mohajer Bidirectional (per budget)
    for b in budgets:
        A19 = cond.get(("dl-2019", "Sampling", "mohajer (ir)", int(b)))
        B19 = cond.get(("dl-2019", "Bidirectional", "mohajer (ir)", int(b)))
        A20 = cond.get(("dl-2020", "Sampling", "mohajer (ir)", int(b)))
        B20 = cond.get(("dl-2020", "Bidirectional", "mohajer (ir)", int(b)))
        if A19 is not None and B19 is not None and A20 is not None and B20 is not None:
            d19 = (A19.align(B19, join="inner")[0] - A19.align(B19, join="inner")[1]).to_numpy(float)
            d20 = (A20.align(B20, join="inner")[0] - A20.align(B20, join="inner")[1]).to_numpy(float)
            stats = paired_bootstrap_delta_macro_dl19_dl20(d19, d20, resamples=resamples, alpha=alpha, rng=rng)
            sig_rows.append(
                {
                    "Table": "table1",
                    "Comparison": "Mohajer_Sampling_vs_Bidirectional",
                    "Oracle": "CrossOracle",
                    "Budget": int(b),
                    "A": "mohajer (ir) [Sampling]",
                    "B": "mohajer (ir) [Bidirectional]",
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
        # percent convenience
        for c in ["delta", "ci_low", "ci_high", "ci_half"]:
            sig_df[c + "_pct"] = 100.0 * sig_df[c]
    sig_df.to_csv(sig_path, index=False)
    print(f"[Table 1] Saved raw runs: {raw_path}")
    print(f"[Table 1] Saved per-query: {query_path}")
    print(f"[Table 1] Saved paired-bootstrap deltas: {sig_path}")


# =========================
# Table 2 runner (end-to-end) + pair tests
# =========================
def run_table2_and_pairs(
    *,
    out_dir: Path,
    matrix_models: List[str],
    datasets: List[str],
    seeds: List[int],
    split: str,
    k: int,
    resamples: int,
    alpha: float,
    bootstrap_seed: int,
    no_resume: bool,
) -> None:
    _ensure_dir(out_dir)
    raw_path = out_dir / "raw_runs.csv"
    query_path = out_dir / "query_ndcg.csv"
    sig_path = out_dir / "paired_bootstrap_pairs.csv"
    avg_path = out_dir / "paired_bootstrap_avg.csv"

    raw_existing = pd.DataFrame() if no_resume else _read_csv(raw_path)
    query_existing = pd.DataFrame() if no_resume else _read_csv(query_path)
    raw_rows = raw_existing.to_dict("records") if not raw_existing.empty else []
    query_rows = query_existing.to_dict("records") if not query_existing.empty else []

    def raw_key(r: dict) -> T2Key:
        return T2Key(
            dataset=str(r["Dataset"]),
            ranker=str(r["Ranker"]),
            oracle=str(r["Oracle"]),
            seed=int(r["Seed"]),
            matrix_model=str(r["MatrixModel"]),
        )

    existing_raw = {raw_key(r) for r in raw_rows} if raw_rows else set()

    def query_key(r: dict) -> T2Key:
        return T2Key(
            dataset=str(r["Dataset"]),
            ranker=str(r["Ranker"]),
            oracle=str(r["Oracle"]),
            seed=int(r["Seed"]),
            matrix_model=str(r["MatrixModel"]),
        )

    existing_query = {query_key(r) for r in query_rows} if query_rows else set()

    def have_run(mm: str, ds: str, rk: str, oc: str, s: int) -> bool:
        k2 = T2Key(ds, rk, oc, int(s), mm)
        return k2 in existing_raw and k2 in existing_query

    # Count missing runs
    total = 0
    for mm in matrix_models:
        for ds in datasets:
            for rk, oc in TABLE2_METHODS:
                for s in seeds:
                    if not have_run(mm, ds, rk, oc, s):
                        total += 1

    pbar = tqdm(total=total, desc="Table 2: runs", unit="run")

    for mm in matrix_models:
        for ds_name in datasets:
            # Some datasets may be missing matrices; skip like run_beir_eval does
            try:
                dataset = load_beir_dataset(ds_name, split=split, matrix_model=mm)
            except FileNotFoundError as e:
                print(f"[Table 2] Skipping {mm}/{ds_name} (missing matrix): {e}")
                continue

            task_qids = [t.query_id for t in dataset.tasks]
            task_count = len(dataset.tasks)

            for s in seeds:
                for rk_name, oc in TABLE2_METHODS:
                    if have_run(mm, ds_name, rk_name, oc, s):
                        continue

                    pbar.set_description(f"T2 {mm}/{ds_name} | {rk_name} | {oc} | S={s}")

                    if oc == "Bidirectional":
                        oracle = BidirectionalMatrixOracle()
                    elif oc == "Sampling":
                        oracle = SamplingMatrixOracle(seed=s)
                    else:
                        raise ValueError(f"Unknown oracle: {oc}")

                    ranker = get_ranker(rk_name, oracle=oracle, seed=s)
                    ranker.set_dataset(ds_name, split=split, query_ids=task_qids, matrix_model=mm)

                    mean_ndcg, comps, avg_comp, perq = evaluate_ranker_with_per_query_ndcg(
                        ranker=ranker, dataset=dataset, seed=s, k=k
                    )

                    raw_rows.append(
                        {
                            "MatrixModel": mm,
                            "Dataset": ds_name,
                            "Ranker": rk_name,
                            "Oracle": oc,
                            "Seed": int(s),
                            "NDCG@10": float(mean_ndcg),
                            "Comparisons": int(comps),
                            "average_comparison_per_task": float(avg_comp),
                        }
                    )
                    existing_raw.add(raw_key(raw_rows[-1]))

                    for qid, v in perq.items():
                        query_rows.append(
                            {
                                "MatrixModel": mm,
                                "Dataset": ds_name,
                                "Ranker": rk_name,
                                "Oracle": oc,
                                "Seed": int(s),
                                "QueryID": str(qid),
                                "NDCG@10_query": float(v),
                            }
                        )
                    existing_query.add(T2Key(ds_name, rk_name, oc, int(s), mm))
                    pbar.update(1)

    pbar.close()

    raw_df = pd.DataFrame(raw_rows).sort_values(
        ["MatrixModel", "Oracle", "Ranker", "Dataset", "Seed"]
    ).reset_index(drop=True)
    query_df = pd.DataFrame(query_rows).sort_values(
        ["MatrixModel", "Oracle", "Ranker", "Dataset", "Seed", "QueryID"]
    ).reset_index(drop=True)

    raw_df.to_csv(raw_path, index=False)
    query_df.to_csv(query_path, index=False)

    # Build seed-averaged per-query vectors:
    # key: (MatrixModel, Dataset, Oracle, Ranker) -> Series indexed by QueryID
    cond = {}
    group_cols = ["MatrixModel", "Dataset", "Oracle", "Ranker"]
    for (mm, ds, oc, rk), g in query_df.groupby(group_cols, sort=True):
        pivot = g.pivot_table(index="QueryID", columns="Seed", values="NDCG@10_query", aggfunc="first")
        per_query_mean = pivot.mean(axis=1, skipna=True)
        cond[(mm, ds, oc, rk)] = per_query_mean

    rng = np.random.default_rng(bootstrap_seed)

    sig_rows = []
    # Per-dataset tests (for table columns)
    for mm in matrix_models:
        for ds in datasets:
            # A: Mohajer sampling, B: BubbleSort bidirectional
            A = cond.get((mm, ds, "Sampling", "mohajer (ir)"))
            B = cond.get((mm, ds, "Bidirectional", "bubble sort (classic)"))
            if A is not None and B is not None and (not A.empty) and (not B.empty):
                a_aligned, b_aligned = A.align(B, join="inner")
                diffs = (a_aligned - b_aligned).to_numpy(float)
                stats = paired_bootstrap_delta(diffs, resamples=resamples, alpha=alpha, rng=rng)
                sig_rows.append(
                    {
                        "Table": "table2",
                        "MatrixModel": mm,
                        "Dataset": ds,
                        "Comparison": "MohajerSampling_vs_BubbleBidirectional",
                        "A": "mohajer (ir) [Sampling]",
                        "B": "bubble sort (classic) [Bidirectional]",
                        "delta": stats["delta"],
                        "ci_low": stats["ci_low"],
                        "ci_high": stats["ci_high"],
                        "ci_half": stats["ci_half"],
                        "p": stats["p"],
                        "n_queries": stats["n"],
                        "sig_p05": bool(np.isfinite(stats["p"]) and stats["p"] < 0.05),
                    }
                )

            # Oracle effect: Mohajer sampling vs Mohajer bidirectional
            A = cond.get((mm, ds, "Sampling", "mohajer (ir)"))
            B = cond.get((mm, ds, "Bidirectional", "mohajer (ir)"))
            if A is not None and B is not None and (not A.empty) and (not B.empty):
                a_aligned, b_aligned = A.align(B, join="inner")
                diffs = (a_aligned - b_aligned).to_numpy(float)
                stats = paired_bootstrap_delta(diffs, resamples=resamples, alpha=alpha, rng=rng)
                sig_rows.append(
                    {
                        "Table": "table2",
                        "MatrixModel": mm,
                        "Dataset": ds,
                        "Comparison": "MohajerSampling_vs_MohajerBidirectional",
                        "A": "mohajer (ir) [Sampling]",
                        "B": "mohajer (ir) [Bidirectional]",
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
    sig_df.to_csv(sig_path, index=False)

    # Optional: macro across datasets for the "Avg" column (unweighted mean over datasets)
    # We do: for each bootstrap replicate, resample queries within each dataset and average deltas across datasets.
    avg_rows = []
    for mm in matrix_models:
        # collect per-dataset diffs arrays for each comparison type
        for comp_name, (A_key, B_key) in [
            (
                "MohajerSampling_vs_BubbleBidirectional",
                (("Sampling", "mohajer (ir)"), ("Bidirectional", "bubble sort (classic)")),
            ),
            (
                "MohajerSampling_vs_MohajerBidirectional",
                (("Sampling", "mohajer (ir)"), ("Bidirectional", "mohajer (ir)")),
            ),
        ]:
            diffs_by_ds = []
            used_ds = []
            for ds in datasets:
                A = cond.get((mm, ds, A_key[0], A_key[1]))
                B = cond.get((mm, ds, B_key[0], B_key[1]))
                if A is None or B is None or A.empty or B.empty:
                    continue
                a_aligned, b_aligned = A.align(B, join="inner")
                diffs = (a_aligned - b_aligned).to_numpy(float)
                diffs = diffs[np.isfinite(diffs)]
                if diffs.size == 0:
                    continue
                diffs_by_ds.append(diffs)
                used_ds.append(ds)

            if not diffs_by_ds:
                continue

            # observed macro delta
            obs = float(np.mean([d.mean() for d in diffs_by_ds]))

            # bootstrap
            boot = np.empty(resamples, dtype=float)
            for i in range(resamples):
                ds_means = []
                for d in diffs_by_ds:
                    n = d.size
                    idx = rng.integers(0, n, size=n)
                    ds_means.append(float(d[idx].mean()))
                boot[i] = float(np.mean(ds_means))

            ci_low = float(np.quantile(boot, alpha / 2))
            ci_high = float(np.quantile(boot, 1 - alpha / 2))
            ci_half = 0.5 * (ci_high - ci_low)
            p = 2 * min(float(np.mean(boot <= 0.0)), float(np.mean(boot >= 0.0)))

            avg_rows.append(
                {
                    "Table": "table2",
                    "MatrixModel": mm,
                    "Comparison": comp_name,
                    "Datasets_used": ",".join(used_ds),
                    "n_datasets": int(len(used_ds)),
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
    avg_df.to_csv(avg_path, index=False)

    print(f"[Table 2] Saved raw runs: {raw_path}")
    print(f"[Table 2] Saved per-query: {query_path}")
    print(f"[Table 2] Saved paired-bootstrap deltas: {sig_path}")
    print(f"[Table 2] Saved paired-bootstrap Avg deltas: {avg_path}")


# =========================
# Entrypoint
# =========================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paired bootstrap significance (A beats B) over queries.")
    p.add_argument("--table", choices=["table1", "table2", "both"], default="both")
    p.add_argument("--no-resume", action="store_true", help="Ignore existing CSVs and rerun everything.")
    p.add_argument("--split", default=DEFAULT_SPLIT)
    p.add_argument("--k", type=int, default=DEFAULT_K)

    p.add_argument("--resamples", type=int, default=10_000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--bootstrap-seed", type=int, default=42)

    p.add_argument("--seeds", type=int, nargs="+", default=[42], help="Seeds to run. Use 42 to reproduce paper means.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    root = Path("reports/significance_testing/paired_bootstrap")
    if args.table in ("table1", "both"):
        run_table1_and_pairs(
            out_dir=root / "table1",
            datasets=DEFAULT_DATASETS_TABLE1,
            budgets=DEFAULT_BUDGETS_TABLE1,
            rankers=DEFAULT_RANKERS_TABLE1,
            seeds=args.seeds,
            matrix_model=DEFAULT_MODEL_TABLE1,
            split=args.split,
            k=args.k,
            resamples=args.resamples,
            alpha=args.alpha,
            bootstrap_seed=args.bootstrap_seed,
            no_resume=args.no_resume,
        )

    if args.table in ("table2", "both"):
        run_table2_and_pairs(
            out_dir=root / "table2",
            matrix_models=DEFAULT_MODELS_TABLE2,
            datasets=DEFAULT_DATASETS_TABLE2,
            seeds=args.seeds,
            split=args.split,
            k=args.k,
            resamples=args.resamples,
            alpha=args.alpha,
            bootstrap_seed=args.bootstrap_seed,
            no_resume=args.no_resume,
        )


if __name__ == "__main__":
    main()
