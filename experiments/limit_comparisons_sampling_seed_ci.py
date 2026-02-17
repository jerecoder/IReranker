import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from ireranker.data.loaders import load_beir_dataset
from ireranker.evaluation.beir import evaluate_rankers_beir
from ireranker.rankers import get_ranker
from ireranker.oracles import BidirectionalMatrixOracle, SamplingMatrixOracle


# =========================
# Experiment Configuration
# =========================
RUN_BIDIRECTIONAL = False
RUN_SAMPLING = True

DATASETS = ["dl-2019", "dl-2020"]
BUDGETS = [100, 150, 200, 250, 300, 350, 400, 450, 500]
RANKERS = [
    "mohajer (ir)",
    "mohajer + bubble",
    "pac + bubble",
    "bubble sort (classic)",
    "quick sort (classic)",
    "heap sort (classic)",
]

# To exactly reproduce the paper table means, set SEEDS = [42]
# To show stability across different runs, use multiple seeds (e.g., 10 seeds below).
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49]

MATRIX_MODEL = "flan-t5-xl"
SPLIT = "test"
K_VALUES = [10]  # evaluate_rankers_beir expects a list

# =========================
# Seed-CI settings
# =========================
SEED_BOOTSTRAP_RESAMPLES = 10_000
CI_ALPHA = 0.05
BOOTSTRAP_RANDOM_SEED = 42

# =========================
# Outputs
# =========================
OUT_DIR = Path("reports/significance_testing")
RAW_BY_SEED_PATH = OUT_DIR / "limit_comparisons_raw_by_seed.csv"

# Dataset-specific seed CI (per dataset, matches your raw CSV layout)
SEED_CI_BY_DATASET_PATH = OUT_DIR / "limit_comparisons_seed_ci_by_dataset.csv"

# Macro DL19+DL20 seed CI (matches how Table 1 is formed: average of DL19 and DL20 means)
SEED_CI_MACRO_TABLE1_PATH = OUT_DIR / "limit_comparisons_seed_ci_macro_dl19_dl20.csv"


def bootstrap_mean_ci_over_seeds(
    values: np.ndarray,
    *,
    num_resamples: int,
    alpha: float,
    rng: np.random.Generator,
):
    """
    Bootstrap CI for the MEAN where the sampling unit is the seed.
    values: shape (n_seeds,)
    Returns: (mean, low, high, halfwidth)
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan
    if n == 1:
        m = float(x[0])
        return m, m, m, 0.0

    point = float(x.mean())
    idx = rng.integers(0, n, size=(num_resamples, n))
    boot = x[idx].mean(axis=1)
    low = float(np.quantile(boot, alpha / 2))
    high = float(np.quantile(boot, 1 - alpha / 2))
    half = 0.5 * (high - low)
    return point, low, high, half


def run_experiment_with_seed_ci():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Resume support for raw-by-seed ----
    if RAW_BY_SEED_PATH.exists():
        print(f"Loading existing results from {RAW_BY_SEED_PATH}")
        raw_df_existing = pd.read_csv(RAW_BY_SEED_PATH)
        raw_rows = raw_df_existing.to_dict("records")
    else:
        raw_rows = []

    def run_key(row: dict):
        return (
            row["Dataset"],
            row["Ranker"],
            row["Oracle"],
            int(row["Budget"]),
            int(row["Seed"]),
        )

    existing_keys = {run_key(r) for r in raw_rows}

    def result_exists(dataset: str, ranker: str, oracle: str, budget: int, seed: int) -> bool:
        return (dataset, ranker, oracle, int(budget), int(seed)) in existing_keys

    # Count remaining runs
    total = 0
    for ds in DATASETS:
        for rk in RANKERS:
            for b in BUDGETS:
                for s in SEEDS:
                    if RUN_BIDIRECTIONAL and not result_exists(ds, rk, "Bidirectional", b, s):
                        total += 1
                    if RUN_SAMPLING and not result_exists(ds, rk, "Sampling", b, s):
                        total += 1

    print(f"Total experiments to run: {total}")
    pbar = tqdm(total=total, desc="Running experiments (seed sweep)")

    for dataset_name in DATASETS:
        print(f"\nProcessing dataset: {dataset_name}")
        dataset = load_beir_dataset(dataset_name, split=SPLIT, matrix_model=MATRIX_MODEL)
        task_qids = [t.query_id for t in dataset.tasks]
        task_count = len(dataset.tasks)

        for seed in SEEDS:
            for ranker_name in RANKERS:
                # 1) Bidirectional
                if RUN_BIDIRECTIONAL:
                    for budget in BUDGETS:
                        if result_exists(dataset_name, ranker_name, "Bidirectional", budget, seed):
                            continue

                        pbar.set_description(
                            f"{dataset_name} | {ranker_name} | Bidirectional | B={budget} | S={seed}"
                        )

                        oracle = BidirectionalMatrixOracle(
                            comparison_limit=budget,
                            comparison_limit_per_task=True,
                        )
                        ranker = get_ranker(ranker_name, oracle=oracle, seed=seed)
                        ranker.set_dataset(
                            dataset_name,
                            split=SPLIT,
                            query_ids=task_qids,
                            matrix_model=MATRIX_MODEL,
                        )

                        metrics = evaluate_rankers_beir([ranker], dataset, K_VALUES, seed=seed)[0]

                        row = {
                            "Dataset": dataset_name,
                            "Ranker": ranker_name,
                            "Oracle": "Bidirectional",
                            "Budget": int(budget),
                            "Seed": int(seed),
                            "NDCG@10": float(metrics["NDCG"]),
                            "Comparisons": float(metrics["Comparisons"]),
                            "average_comparison_per_task": float(metrics["Comparisons_per_task"])
                            if "Comparisons_per_task" in metrics
                            else float(metrics["Comparisons"]) / task_count if task_count else 0.0,
                        }
                        raw_rows.append(row)
                        existing_keys.add(run_key(row))
                        pbar.update(1)

                # 2) Sampling / Randomized-direction
                if RUN_SAMPLING:
                    for budget in BUDGETS:
                        if result_exists(dataset_name, ranker_name, "Sampling", budget, seed):
                            continue

                        pbar.set_description(
                            f"{dataset_name} | {ranker_name} | Sampling | B={budget} | S={seed}"
                        )

                        oracle = SamplingMatrixOracle(
                            seed=seed,
                            comparison_limit=budget,
                            comparison_limit_per_task=True,
                        )
                        ranker = get_ranker(ranker_name, oracle=oracle, seed=seed)
                        ranker.set_dataset(
                            dataset_name,
                            split=SPLIT,
                            query_ids=task_qids,
                            matrix_model=MATRIX_MODEL,
                        )

                        metrics = evaluate_rankers_beir([ranker], dataset, K_VALUES, seed=seed)[0]

                        row = {
                            "Dataset": dataset_name,
                            "Ranker": ranker_name,
                            "Oracle": "Sampling",
                            "Budget": int(budget),
                            "Seed": int(seed),
                            "NDCG@10": float(metrics["NDCG"]),
                            "Comparisons": float(metrics["Comparisons"]),
                            "average_comparison_per_task": float(metrics["Comparisons_per_task"])
                            if "Comparisons_per_task" in metrics
                            else float(metrics["Comparisons"]) / task_count if task_count else 0.0,
                        }
                        raw_rows.append(row)
                        existing_keys.add(run_key(row))
                        pbar.update(1)

    pbar.close()

    raw_df = pd.DataFrame(raw_rows)
    raw_df = raw_df.sort_values(["Oracle", "Ranker", "Dataset", "Budget", "Seed"]).reset_index(drop=True)
    raw_df.to_csv(RAW_BY_SEED_PATH, index=False)
    print(f"Saved raw-by-seed results to {RAW_BY_SEED_PATH}")

    # ============================================================
    # Seed CI summaries
    # ============================================================
    rng = np.random.default_rng(BOOTSTRAP_RANDOM_SEED)

    # ---- (A) Dataset-level CI: (Dataset, Oracle, Ranker, Budget) over seeds ----
    ci_rows = []
    for (ds, oracle, ranker, budget), g in raw_df.groupby(
        ["Dataset", "Oracle", "Ranker", "Budget"], sort=True
    ):
        vals = g["NDCG@10"].to_numpy(float)
        point, low, high, half = bootstrap_mean_ci_over_seeds(
            vals,
            num_resamples=SEED_BOOTSTRAP_RESAMPLES,
            alpha=CI_ALPHA,
            rng=rng,
        )
        ci_rows.append(
            {
                "Dataset": ds,
                "Oracle": oracle,
                "Ranker": ranker,
                "Budget": int(budget),
                "seeds_used": int(g["Seed"].nunique()),
                "ndcg_mean": point,
                "ci_low": low,
                "ci_high": high,
                "ci_half": half,
                "bootstrap_resamples": int(SEED_BOOTSTRAP_RESAMPLES),
                "alpha": float(CI_ALPHA),
            }
        )

    ci_df = pd.DataFrame(ci_rows).sort_values(
        ["Oracle", "Ranker", "Dataset", "Budget"]
    ).reset_index(drop=True)

    for c in ["ndcg_mean", "ci_low", "ci_high", "ci_half"]:
        ci_df[c + "_pct"] = 100.0 * ci_df[c]

    ci_df.to_csv(SEED_CI_BY_DATASET_PATH, index=False)
    print(f"Saved dataset-level seed CIs to {SEED_CI_BY_DATASET_PATH}")

    # ---- (B) Macro Table-1 CI: average of DL19 and DL20 means per seed, then CI over seeds ----
    # Build per-seed macro values first:
    macro_seed = (
        raw_df[raw_df["Dataset"].isin(["dl-2019", "dl-2020"])]
        .groupby(["Oracle", "Ranker", "Budget", "Seed"], as_index=False)["NDCG@10"]
        .mean()
        .rename(columns={"NDCG@10": "NDCG@10_macro_dl19_dl20"})
    )

    macro_rows = []
    for (oracle, ranker, budget), g in macro_seed.groupby(["Oracle", "Ranker", "Budget"], sort=True):
        vals = g["NDCG@10_macro_dl19_dl20"].to_numpy(float)
        point, low, high, half = bootstrap_mean_ci_over_seeds(
            vals,
            num_resamples=SEED_BOOTSTRAP_RESAMPLES,
            alpha=CI_ALPHA,
            rng=rng,
        )
        macro_rows.append(
            {
                "Oracle": oracle,
                "Ranker": ranker,
                "Budget": int(budget),
                "seeds_used": int(g["Seed"].nunique()),
                "ndcg_mean_macro": point,
                "ci_low": low,
                "ci_high": high,
                "ci_half": half,
                "bootstrap_resamples": int(SEED_BOOTSTRAP_RESAMPLES),
                "alpha": float(CI_ALPHA),
            }
        )

    macro_ci_df = pd.DataFrame(macro_rows).sort_values(["Oracle", "Ranker", "Budget"]).reset_index(drop=True)
    for c in ["ndcg_mean_macro", "ci_low", "ci_high", "ci_half"]:
        macro_ci_df[c + "_pct"] = 100.0 * macro_ci_df[c]

    macro_ci_df.to_csv(SEED_CI_MACRO_TABLE1_PATH, index=False)
    print(f"Saved Table-1-style macro seed CIs to {SEED_CI_MACRO_TABLE1_PATH}")

    print("\nDone.")
    print("If you set SEEDS=[42], ndcg_mean_macro_pct should match your paper Table 1 values.")
    print("If you use multiple seeds, ndcg_mean_macro_pct becomes the seed-averaged mean,")
    print("and ci_*_pct gives the 95% CI of that mean over seed variability.")


if __name__ == "__main__":
    run_experiment_with_seed_ci()
