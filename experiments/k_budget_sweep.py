"""
K and Budget Sweep Experiment

Sweeps both K values (10, 20, 30, 40, 50) and comparison limits for multiple rankers.
Outputs a 2D grid of NDCG@K values.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ireranker.data.loaders import load_beir_dataset
from ireranker.evaluation.beir import evaluate_rankers_beir
from ireranker.rankers import get_ranker
from ireranker.oracles import SamplingMatrixOracle


def run_experiment():
    # Experiment Configuration
    datasets = ["dl-2019", "dl-2020"]
    ranker_names = ["mohajer (ir)", "bubble sort (classic)", "bm25"]
    k_values = [10, 20, 30, 40, 50]
    budgets = [100, 150, 200, 250, 300, 350, 400, 450, 500]
    
    seed = 42
    matrix_model = "flan-t5-xl"
    
    results = []
    output_path = Path("reports/k_budget_sweep.csv")
    
    # Load existing results if available
    if output_path.exists():
        print(f"Loading existing results from {output_path}")
        df = pd.read_csv(output_path)
        results = df.to_dict('records')
    
    # Helper to check if result exists
    def result_exists(dataset, ranker, k, budget):
        for r in results:
            if (r["Dataset"] == dataset and 
                r["Ranker"] == ranker and
                r["K"] == k and 
                r["Budget"] == budget):
                return True
        return False

    # Calculate total experiments for progress bar
    total_experiments = 0
    for dataset_name in datasets:
        for ranker_name in ranker_names:
            for k in k_values:
                for budget in budgets:
                    if not result_exists(dataset_name, ranker_name, k, budget):
                        total_experiments += 1

    print(f"Total experiments to run: {total_experiments}")
    pbar = tqdm(total=total_experiments, desc="Running experiments")

    for dataset_name in datasets:
        print(f"\nProcessing dataset: {dataset_name}")
        dataset = load_beir_dataset(dataset_name, split="test", matrix_model=matrix_model)
        task_qids = [t.query_id for t in dataset.tasks]

        for ranker_name in ranker_names:
            for k in k_values:
                for budget in budgets:
                    if result_exists(dataset_name, ranker_name, k, budget):
                        continue

                    pbar.set_description(f"{dataset_name} | {ranker_name} | K={k} | Budget={budget}")
                    
                    oracle = SamplingMatrixOracle(
                        seed=seed,
                        comparison_limit=budget, 
                        comparison_limit_per_task=True
                    )
                    ranker = get_ranker(ranker_name, oracle=oracle, seed=seed)
                    ranker.set_dataset(
                        dataset_name,
                        split="test",
                        query_ids=task_qids,
                        matrix_model=matrix_model,
                    )

                    # Evaluate for this specific K
                    metrics = evaluate_rankers_beir([ranker], dataset, [k], seed=seed)[0]

                    results.append({
                        "Dataset": dataset_name,
                        "Ranker": ranker_name,
                        "Oracle": "Sampling",
                        "K": k,
                        "Budget": budget,
                        "NDCG": metrics["NDCG"],
                        "Comparisons": metrics["Comparisons"],
                    })
                    
                    # Save after each result for safety
                    df = pd.DataFrame(results)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(output_path, index=False)
                    
                    pbar.update(1)

    pbar.close()

    # Final save
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    # --- Generate README table ---
    readme_path = Path("README.md")
    if readme_path.exists():
        readme_content = readme_path.read_text(encoding="utf-8")
        
        header = "## K Budget Sweep Experiment"
        if header in readme_content:
            readme_content = readme_content.split(header)[0].rstrip()
        
        new_section = [header]
        new_section.append(f"\nRankers: **{', '.join(ranker_names)}** with Sampling Oracle\n")
        
        for dataset in datasets:
            ds_df = df[df["Dataset"] == dataset]
            if ds_df.empty:
                continue

            new_section.append(f"### {dataset}")
            
            for ranker_name in ranker_names:
                ranker_df = ds_df[ds_df["Ranker"] == ranker_name]
                if ranker_df.empty:
                    continue
                
                new_section.append(f"\n#### {ranker_name}")
                
                # Pivot: Rows = K, Columns = Budget, Values = NDCG
                pivot_df = ranker_df.pivot_table(
                    index="K", 
                    columns="Budget", 
                    values="NDCG", 
                    aggfunc='first'
                )
            pivot_df = pivot_df.reset_index()
            columns = pivot_df.columns.tolist()
            
            # Header row
            header_row = "| " + " | ".join(str(c) for c in columns) + " |"
            new_section.append(header_row)
            divider_row = "| " + " | ".join(["---"] * len(columns)) + " |"
            new_section.append(divider_row)
            
            # Find max per column for bolding
            max_per_col = {}
            for col in columns:
                if col == "K":
                    continue
                max_per_col[col] = pivot_df[col].max()

            # Data rows
            for _, row in pivot_df.iterrows():
                formatted_values = []
                for val, col_name in zip(row.values, columns):
                    if col_name == "K":
                        formatted_values.append(str(int(val)))
                        continue

                    if pd.notna(val):
                        str_val = f"{val:.4f}"
                        if np.isclose(val, max_per_col[col_name]):
                            str_val = f"**{str_val}**"
                    else:
                        str_val = "-"
                    formatted_values.append(str_val)
                new_section.append("| " + " | ".join(formatted_values) + " |")
                new_section.append("")

        new_content = readme_content + "\n\n" + "\n".join(new_section) + "\n"
        readme_path.write_text(new_content, encoding="utf-8")
        print(f"Results written to {readme_path}")


if __name__ == "__main__":
    run_experiment()
