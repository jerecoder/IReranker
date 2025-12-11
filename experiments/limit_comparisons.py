
import pandas as pd
import numpy as np
from pathlib import Path
from ireranker.data.loaders import load_beir_dataset
from ireranker.evaluation.beir import evaluate_rankers_beir
from ireranker.rankers import get_ranker
from ireranker.oracles import BidirectionalMatrixOracle, SamplingMatrixOracle

def run_experiment():
    datasets = ["trec-covid", "dbpedia-entity", "fiqa"]
    budgets = [50, 100, 150, 200, 250, 300, 500]
    seeds = [42]
    matrix_model = "flan-t5-large"
    k_values = [10]
    
    results = []
    output_path = Path("reports/limit_comparisons_experiment.csv")

    ranker_names = ["Mohajer (IR)", "Bubble Sort (Classic)", "Quick Sort (Classic)", "Mohajer + Bubble", "PAC + Bubble"]
    
    # Check if results exist to avoid re-running
    if output_path.exists():
        print(f"Loading existing results from {output_path}")
        df = pd.read_csv(output_path)
        if "Budget" not in df.columns:
            print("Inferring 'Budget' column from data order...")
            # Structure: Datasets (3) * Rankers (3) * Oracles (2) * Budgets (7)
            # Total groups = 3 * 3 * 2 = 18
            # Total rows should be 18 * 7 = 126
            expected_rows = len(datasets) * len(ranker_names) * 2 * len(budgets)
            if len(df) == expected_rows:
                # The budgets list repeats for every group
                df["Budget"] = budgets * (len(df) // len(budgets))
            else:
                print(f"Warning: Row count {len(df)} does not match expected {expected_rows}. Cannot safely infer budgets. Re-running experiment.")
                output_path.unlink() # Force re-run
                results = []
    
    if not output_path.exists():
        results = []
        for dataset_name in datasets:
            print(f"Processing dataset: {dataset_name}")
            dataset = load_beir_dataset(dataset_name, split="test", matrix_model=matrix_model)
            task_qids = [t.query_id for t in dataset.tasks]

            for seed in seeds:
                for ranker_name in ranker_names:
                    # 1. Bidirectional Oracle
                    for budget in budgets:
                        print(f"  {ranker_name}, Bidirectional Oracle, Budget: {budget}")
                        oracle = BidirectionalMatrixOracle(comparison_limit=budget, comparison_limit_per_task=True)
                        ranker = get_ranker(ranker_name, oracle=oracle, seed=seed)
                        ranker.set_dataset(
                            dataset_name,
                            split="test",
                            query_ids=task_qids,
                            matrix_model=matrix_model,
                        )
                        
                        metrics = evaluate_rankers_beir([ranker], dataset, k_values, seed=seed)[0]
                        
                        results.append({
                            "Ranker": f"{ranker_name} [Bidirectional]",
                            "Dataset": dataset_name,
                            "Comparisons": metrics["Comparisons"],
                            "NDCG@10": metrics["NDCG"],
                            "Budget": budget
                        })

                    # 2. Sampling Oracle
                    for budget in budgets:
                        print(f"  {ranker_name}, Sampling Oracle, Budget: {budget}")
                        oracle = SamplingMatrixOracle(seed=seed, comparison_limit=budget, comparison_limit_per_task=True)
                        ranker = get_ranker(ranker_name, oracle=oracle, seed=seed)
                        ranker.set_dataset(
                            dataset_name,
                            split="test",
                            query_ids=task_qids,
                            matrix_model=matrix_model,
                        )
                        
                        metrics = evaluate_rankers_beir([ranker], dataset, k_values, seed=seed)[0]
                        
                        results.append({
                            "Ranker": f"{ranker_name} [Sampling]",
                            "Dataset": dataset_name,
                            "Comparisons": metrics["Comparisons"],
                            "NDCG@10": metrics["NDCG"],
                            "Budget": budget
                        })

        df = pd.DataFrame(results)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

    # Update README
    readme_path = Path("README.md")
    if readme_path.exists():
        readme_content = readme_path.read_text(encoding="utf-8")
        
        # Remove existing "Limit Comparisons Experiment" sections
        header = "## Limit Comparisons Experiment"
        if header in readme_content:
            readme_content = readme_content.split(header)[0].rstrip()
        
        new_section = [header]
        
        # Determine unique datasets to loop over, preserving order
        unique_datasets = df["Dataset"].unique()
        
        for dataset in unique_datasets:
            new_section.append(f"\n### {dataset}")
            
            # Filter for this dataset
            ds_df = df[df["Dataset"] == dataset]
            
            # Pivot: Index=Ranker, Columns=Budget, Values=NDCG@10
            pivot_df = ds_df.pivot(index="Ranker", columns="Budget", values="NDCG@10")
            
            # Reset index to make Ranker a column for printing
            pivot_df = pivot_df.reset_index()
            
            # Manual Markdown Table Generation
            columns = pivot_df.columns.tolist() # ['Ranker', 100, 1000, ...]
            
            # Header Row
            header_row = "| " + " | ".join(str(c) for c in columns) + " |"
            new_section.append(header_row)
            
            # Divider Row
            divider_row = "| " + " | ".join(["---"] * len(columns)) + " |"
            new_section.append(divider_row)
            
            # Data Rows
            # Data Rows
            # Find max per column (excluding 'Ranker')
            max_per_col = {}
            for col in columns:
                if col == "Ranker":
                    continue
                # pivot_df[col] might handle types weirdly if mixed. 
                # Values are floats.
                max_per_col[col] = pivot_df[col].max()

            for _, row in pivot_df.iterrows():
                # Format floats to 4 decimal places if possible
                formatted_values = []
                for val, col_name in zip(row.values, columns):
                    if col_name == "Ranker":
                         formatted_values.append(str(val))
                         continue

                    is_max = False
                    if isinstance(val, (float, np.floating)):
                         if np.isclose(val, max_per_col[col_name]):
                             is_max = True
                         str_val = f"{val:.4f}"
                    else:
                         if val == max_per_col[col_name]:
                             is_max = True
                         str_val = str(val)
                    
                    if is_max:
                        str_val = f"**{str_val}**"
                    formatted_values.append(str_val)
                row_str = "| " + " | ".join(formatted_values) + " |"
                new_section.append(row_str)
        
        new_content = readme_content + "\n\n" + "\n".join(new_section) + "\n"
        
        readme_path.write_text(new_content, encoding="utf-8")
        print(f"Results written to {readme_path} (overwriting previous table)")
    else:
        print(f"README.md not found at {readme_path}")
if __name__ == "__main__":
    run_experiment()
