
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ireranker.data.loaders import load_beir_dataset
from ireranker.evaluation.beir import evaluate_rankers_beir
from ireranker.rankers import get_ranker
from ireranker.oracles import BidirectionalMatrixOracle, SamplingMatrixOracle

def run_experiment():
    # All datasets from config
    datasets = [
        "webis-touche2020",
        "trec-covid",
        "scifact",
        "fiqa",
        "dbpedia-entity",
        "dl-2019",
        "dl-2020",
        "robust04"
    ]

    # Doubled step size for faster completion
    budgets = [50, 100, 200, 400, 800]
    rankers = [
        "mohajer (ir)",
        "mohajer + bubble",
        "pac + bubble",
        "bubble sort (classic)",
        "quick sort (classic)",
        "prp sort (classic)"
    ]

    seeds = [42]
    matrix_model = "flan-t5-large"
    k_values = [10]
    
    results = []
    output_path = Path("reports/limit_comparisons_experiment.csv")
    
    # We will simply overwrite or append. Given the complexity of splitting, 
    # let's just re-run fresh if the user wants clarity, or we could handle existing results.
    # For now, let's keep it simple: if file exists, warn/load? 
    # Actually, simpler to just run and overwrite if the user wants to re-run.
    # But usually we check for existing to avoid re-computation.
    # Let's drop the complicated inference logic for now since structure changed.
    if output_path.exists():
        print(f"Loading existing results from {output_path}")
        df = pd.read_csv(output_path)
        # We assume the columns are correct.
        results = df.to_dict('records')
    else:
        results = []

    # Helper to check if result exists
    def result_exists(dataset, ranker, budget, oracle_type):
        for r in results:
            if (r["Dataset"] == dataset and 
                r["Ranker"] == ranker and 
                r["Budget"] == budget and
                oracle_type in r["Oracle"]):
                return True
        return False

    # Calculate total experiments for progress bar
    total_experiments = 0
    for dataset_name in datasets:
        for ranker_name in rankers:
            for budget in budgets:
                if not result_exists(dataset_name, ranker_name, budget, "Bidirectional"):
                    total_experiments += 1
                if not result_exists(dataset_name, ranker_name, budget, "Sampling"):
                    total_experiments += 1

    print(f"Total experiments to run: {total_experiments}")

    pbar = tqdm(total=total_experiments, desc="Running experiments")

    for dataset_name in datasets:
        print(f"\nProcessing dataset: {dataset_name}")
        dataset = load_beir_dataset(dataset_name, split="test", matrix_model=matrix_model)
        task_qids = [t.query_id for t in dataset.tasks]

        for seed in seeds:
            for ranker_name in rankers:
                # 1. Bidirectional Oracle
                for budget in budgets:
                    ranker_label = f"{ranker_name} [Bidirectional]"
                    if result_exists(dataset_name, ranker_name, budget, "Bidirectional"):
                        continue

                    pbar.set_description(f"{dataset_name} | {ranker_name} | Bidirectional | Budget: {budget}")
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
                        "Ranker": ranker_name,
                        "Oracle": "Bidirectional",
                        "DisplayRanker": f"{ranker_name} [Bidirectional]",
                        "Dataset": dataset_name,
                        "Comparisons": metrics["Comparisons"],
                        "NDCG@10": metrics["NDCG"],
                        "Budget": budget
                    })
                    pbar.update(1)

                # 2. Sampling Oracle
                for budget in budgets:
                    ranker_label = f"{ranker_name} [Sampling]"
                    if result_exists(dataset_name, ranker_name, budget, "Sampling"):
                        continue

                    pbar.set_description(f"{dataset_name} | {ranker_name} | Sampling | Budget: {budget}")
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
                        "Ranker": ranker_name,
                        "Oracle": "Sampling",
                        "DisplayRanker": f"{ranker_name} [Sampling]",
                        "Dataset": dataset_name,
                        "Comparisons": metrics["Comparisons"],
                        "NDCG@10": metrics["NDCG"],
                        "Budget": budget
                    })
                    pbar.update(1)

    pbar.close()

    # Save Results
    df = pd.DataFrame(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    # --- Update README ---
    readme_path = Path("README.md")
    if readme_path.exists():
        readme_content = readme_path.read_text(encoding="utf-8")
        
        # Remove existing "Limit Comparisons Experiment" sections
        # We need a robust way to replace the whole block.
        header = "## Limit Comparisons Experiment"
        if header in readme_content:
            readme_content = readme_content.split(header)[0].rstrip()
        
        new_section = [header]
        
        unique_datasets = df["Dataset"].unique()

        for dataset in unique_datasets:
            ds_df = df[df["Dataset"] == dataset]
            if ds_df.empty:
                continue

            new_section.append(f"\n### {dataset}")
            
            # Pivot: Index=DisplayRanker, Columns=Budget, Values=NDCG@10
            pivot_df = ds_df.pivot_table(index="DisplayRanker", columns="Budget", values="NDCG@10", aggfunc='first')
            
            # Sort indices: grouping by Oracle type helps readability
            pivot_df = pivot_df.sort_index()
            
            pivot_df = pivot_df.reset_index()
            columns = pivot_df.columns.tolist() 
            
            # Header
            header_row = "| " + " | ".join(str(c) for c in columns) + " |"
            new_section.append(header_row)
            divider_row = "| " + " | ".join(["---"] * len(columns)) + " |"
            new_section.append(divider_row)
            
            # Max per column
            max_per_col = {}
            for col in columns:
                if col == "DisplayRanker": continue
                max_per_col[col] = pivot_df[col].max()

            # Rows
            for _, row in pivot_df.iterrows():
                formatted_values = []
                for val, col_name in zip(row.values, columns):
                    if col_name == "DisplayRanker":
                         formatted_values.append(str(val))
                         continue

                    is_max = False
                    if pd.notna(val): # Handle NaNs if budget missing for some ranker
                        if isinstance(val, (float, np.floating)):
                            if np.isclose(val, max_per_col[col_name]):
                                is_max = True
                            str_val = f"{val:.4f}"
                        else:
                            if val == max_per_col[col_name]:
                                is_max = True
                            str_val = str(val)
                    else:
                        str_val = "-"
                    
                    if is_max:
                        str_val = f"**{str_val}**"
                    formatted_values.append(str_val)
                new_section.append("| " + " | ".join(formatted_values) + " |")

        new_content = readme_content + "\n\n" + "\n".join(new_section) + "\n"
        readme_path.write_text(new_content, encoding="utf-8")
        print(f"Results written to {readme_path}")

if __name__ == "__main__":
    run_experiment()
