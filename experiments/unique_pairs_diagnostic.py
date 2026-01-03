"""
Unique Pairs per Budget Diagnostic Experiment

Logs every pairwise query (i,j) during reranking and computes:
1. Number of distinct unordered pairs queried
2. Re-query rate (% of calls repeating already-seen pairs)

Tests the hypothesis that randomized-direction oracle improvements come from 
spending the same call budget on more unique evidence rather than duplicating comparisons.
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm
from ireranker.data.loaders import load_beir_dataset
from ireranker.evaluation.beir import evaluate_rankers_beir
from ireranker.rankers import get_ranker
from ireranker.oracles import BidirectionalMatrixOracle, SamplingMatrixOracle, TrackingOracle


def run_experiment():
    """Run unique pairs diagnostic experiment."""
    
    # Configuration
    datasets = ["dl-2019", "dl-2020"]
    budgets = [150, 200, 350]
    rankers_config = [
        "mohajer (ir)",
        "bubble sort (classic)",
        "quick sort (classic)",
        "prp sort (classic)",
    ]
    
    oracle_configs = [
        ("Bidirectional", lambda seed, budget: BidirectionalMatrixOracle(
            comparison_limit=budget, comparison_limit_per_task=True
        )),
        ("Sampling", lambda seed, budget: SamplingMatrixOracle(
            seed=seed, comparison_limit=budget, comparison_limit_per_task=True
        )),
    ]
    
    seed = 42
    matrix_model = "flan-t5-xl"
    k_values = [10]
    
    results = []
    output_path = Path("reports/unique_pairs_diagnostic.csv")
    
    # Calculate total experiments
    total_experiments = len(datasets) * len(rankers_config) * len(oracle_configs) * len(budgets)
    print(f"Total experiment runs: {total_experiments}")
    print(f"Budgets: {budgets}")
    
    pbar = tqdm(total=total_experiments, desc="Running experiments")
    
    for dataset_name in datasets:
        print(f"\nProcessing dataset: {dataset_name}")
        dataset = load_beir_dataset(dataset_name, split="test", matrix_model=matrix_model)
        task_qids = [t.query_id for t in dataset.tasks]
        
        for ranker_name in rankers_config:
            for oracle_label, oracle_factory in oracle_configs:
                for budget in budgets:
                    pbar.set_description(f"{dataset_name} | {ranker_name} | {oracle_label} | B={budget}")
                    
                    # Create inner oracle with budget limit and wrap with tracking
                    inner_oracle = oracle_factory(seed, budget)
                    tracking_oracle = TrackingOracle(inner_oracle)
                    
                    # Get ranker with tracking oracle
                    ranker = get_ranker(ranker_name, oracle=tracking_oracle, seed=seed)
                    ranker.set_dataset(
                        dataset_name,
                        split="test",
                        query_ids=task_qids,
                        matrix_model=matrix_model,
                    )
                    
                    # Run evaluation
                    metrics = evaluate_rankers_beir([ranker], dataset, k_values, seed=seed)[0]
                    
                    # Finalize tracking and get aggregated stats
                    tracking_oracle.finalize_current_task()
                    stats = tracking_oracle.get_aggregated_stats_at_budget(budget)
                    
                    results.append({
                        "Dataset": dataset_name,
                        "Ranker": ranker_name,
                        "Oracle": oracle_label,
                        "Budget": budget,
                        "NumTasks": stats["num_tasks"],
                        "TotalCalls": stats["total_calls"],
                        "DistinctPairs": stats["distinct_pairs"],
                        "RequeryCount": stats["requery_count"],
                        "RequeryRate": round(stats["requery_rate"] * 100, 2),  # as percentage
                        "NDCG@10": round(metrics["NDCG"], 4),
                    })
                    
                    pbar.update(1)
    
    pbar.close()
    
    # Save results
    df = pd.DataFrame(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    
    # Print summary tables
    print_summary(df, datasets, budgets)
    
    return df


def print_summary(df: pd.DataFrame, datasets: list, budgets: list):
    """Print formatted summary tables."""
    print("\n" + "=" * 80)
    print("UNIQUE PAIRS DIAGNOSTIC SUMMARY")
    print("=" * 80)
    
    for dataset in datasets:
        print(f"\n### {dataset}")
        ds_df = df[df["Dataset"] == dataset]
        
        for budget in budgets:
            print(f"\n**Budget = {budget}**")
            budget_df = ds_df[ds_df["Budget"] == budget].copy()
            
            # Create readable display
            budget_df["DisplayName"] = budget_df["Ranker"] + " [" + budget_df["Oracle"] + "]"
            display_df = budget_df[["DisplayName", "TotalCalls", "DistinctPairs", "RequeryRate", "NDCG@10"]]
            display_df = display_df.set_index("DisplayName")
            print(display_df.to_string())
    
    # Comparative summary: Bidirectional vs Sampling
    print("\n" + "=" * 80)
    print("BIDIRECTIONAL vs SAMPLING COMPARISON")
    print("=" * 80)
    
    comparison = []
    for (dataset, ranker, budget), group in df.groupby(["Dataset", "Ranker", "Budget"]):
        bidir = group[group["Oracle"] == "Bidirectional"]
        sampl = group[group["Oracle"] == "Sampling"]
        
        if not bidir.empty and not sampl.empty:
            comparison.append({
                "Dataset": dataset,
                "Ranker": ranker,
                "Budget": budget,
                "Bidir_DistinctPairs": bidir["DistinctPairs"].values[0],
                "Sampl_DistinctPairs": sampl["DistinctPairs"].values[0],
                "Bidir_RequeryRate%": bidir["RequeryRate"].values[0],
                "Sampl_RequeryRate%": sampl["RequeryRate"].values[0],
                "Bidir_NDCG": bidir["NDCG@10"].values[0],
                "Sampl_NDCG": sampl["NDCG@10"].values[0],
            })
    
    comp_df = pd.DataFrame(comparison)
    comp_df["DistinctPairs_Diff"] = comp_df["Sampl_DistinctPairs"] - comp_df["Bidir_DistinctPairs"]
    comp_df["RequeryRate_Diff"] = comp_df["Bidir_RequeryRate%"] - comp_df["Sampl_RequeryRate%"]
    comp_df["NDCG_Diff"] = comp_df["Sampl_NDCG"] - comp_df["Bidir_NDCG"]
    
    print("\nSummary: Sampling vs Bidirectional differences")
    print("(Positive DistinctPairs_Diff = Sampling queries more unique pairs)")
    print("(Positive RequeryRate_Diff = Bidirectional re-queries more)")
    print("(Positive NDCG_Diff = Sampling performs better)")
    print()
    print(comp_df.to_string(index=False))


if __name__ == "__main__":
    run_experiment()
