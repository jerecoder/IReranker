
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def main():
    # Config
    DATA_PATH = Path("reports/limit_comparisons_experiment.csv")
    OUTPUT_DIR = Path("reports")

    # Define colors
    colors_map = {
        'Mohajer (IR)': '#345834',
        'PAC + Bubble': '#65976a',
        'Bubble Sort (Classic)': '#791818',
        'Quick Sort (Classic)': '#dfa51c',
        'PRP Sort (classic)': '#d51d1b',
        'Jingle Bells': '#1b4dd5',
        'Christmas Tree': '#8b2ed5',
        'bm25': '#444444',
        'BM25': '#444444',
    }

    # Verify file exists
    if not DATA_PATH.exists():
        print(f"Error: {DATA_PATH} not found.")
        return

    # Load Data
    df = pd.read_csv(DATA_PATH)
    
    # Filter Budget <= 700
    df = df[df["Budget"] <= 700].copy()
    
    sns.set_style("whitegrid")
    
    datasets = df["Dataset"].unique()
    
    # --- Definition of Ranker/Oracle Sets ---
    
    # Set 1: Main Comparison
    main_comparison_pairs = [
        ("Mohajer (IR)", "Sampling"), 
        ("PAC + Bubble", "Sampling"), 
        ("Bubble Sort (Classic)", "Bidirectional"),
        ("Quick Sort (Classic)", "Bidirectional"),
        ("PRP Sort (classic)", "Bidirectional"),
        ("Jingle Bells", "Sampling"),
        ("Christmas Tree", "Sampling"),
        ("bm25", "Bidirectional"),
    ]
    
    # Set 2: Oracle Comparison (Only Baselines)
    oracle_comparison_rankers = [
        "Mohajer (IR)",
        "PAC + Bubble",
        "Bubble Sort (Classic)",
        "Quick Sort (Classic)",
        "PRP Sort (classic)",
        "Jingle Bells",
        "Christmas Tree",
        "bm25",
    ]

    def _mask_for_pairs(dataframe, pairs):
        mask = pd.Series([False] * len(dataframe), index=dataframe.index)
        for ranker, oracle in pairs:
            m = (dataframe["Ranker"] == ranker) & (dataframe["Oracle"] == oracle)
            mask = mask | m
        return mask

    for ds in datasets:
        ds_data = df[df["Dataset"] == ds]
        if ds_data.empty: continue
        
        # ---------------------------------------------------------
        # Plot 1: Main Comparison (Baselines[Bi] vs Mohajer/PAC[Sa])
        # ---------------------------------------------------------
        mask1 = _mask_for_pairs(ds_data, main_comparison_pairs)
        data1 = ds_data[mask1].sort_values("Budget")
        
        if not data1.empty:
            plt.figure(figsize=(10, 6))
            sns.lineplot(
                data=data1, 
                x="Budget", 
                y="NDCG@10", 
                hue="Ranker", 
                palette=colors_map, 
                marker="o",
                linewidth=2
            )
            plt.title(f"{ds}")
            plt.xlabel("Comparison Budget")
            plt.ylabel("NDCG@10")
            plt.xlim(0, 750)
            plt.tight_layout()
            
            out1 = OUTPUT_DIR / f"limit_comparisons_main_{ds}.png"
            plt.savefig(out1)
            print(f"Saved {out1}")
            plt.close()

    # ---------------------------------------------------------
    # Plot 2: Oracle Comparison (Baselines: Bi vs Sa)
    # ---------------------------------------------------------
        data2 = ds_data[ds_data["Ranker"].isin(oracle_comparison_rankers)].sort_values("Budget")
        
        if not data2.empty:
            plt.figure(figsize=(10, 6))
            sns.lineplot(
                data=data2, 
                x="Budget", 
                y="NDCG@10", 
                hue="Ranker", 
                style="Oracle",
                palette=colors_map, 
                markers=True,
                dashes={"Bidirectional": "", "Sampling": (2, 2)},
                linewidth=2
            )
            plt.title(f"{ds}")
            plt.xlabel("Comparison Budget")
            plt.ylabel("NDCG@10")
            plt.xlim(0, 750)
            plt.tight_layout()
            
            out2 = OUTPUT_DIR / f"limit_comparisons_oracles_{ds}.png"
            plt.savefig(out2)
            print(f"Saved {out2}")
            plt.close()

    # ---------------------------------------------------------
    # Aggregated plots across all datasets (average NDCG@10)
    # ---------------------------------------------------------
    agg_main = df[_mask_for_pairs(df, main_comparison_pairs)]
    if not agg_main.empty:
        grouped = (
            agg_main.groupby(["Ranker", "Oracle", "Budget"], as_index=False)["NDCG@10"]
            .mean()
            .sort_values("Budget")
        )
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=grouped,
            x="Budget",
            y="NDCG@10",
            hue="Ranker",
            style="Oracle",
            palette=colors_map,
            markers=True,
            linewidth=2,
        )
        plt.title("All datasets (avg NDCG@10) — Main comparison")
        plt.xlabel("Comparison Budget")
        plt.ylabel("Avg NDCG@10")
        plt.xlim(0, 750)
        plt.tight_layout()
        out_all_main = OUTPUT_DIR / "limit_comparisons_main_all.png"
        plt.savefig(out_all_main)
        print(f"Saved {out_all_main}")
        plt.close()

    agg_oracle = df[df["Ranker"].isin(oracle_comparison_rankers)]
    if not agg_oracle.empty:
        grouped = (
            agg_oracle.groupby(["Ranker", "Oracle", "Budget"], as_index=False)["NDCG@10"]
            .mean()
            .sort_values("Budget")
        )
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=grouped,
            x="Budget",
            y="NDCG@10",
            hue="Ranker",
            style="Oracle",
            palette=colors_map,
            markers=True,
            dashes={"Bidirectional": "", "Sampling": (2, 2)},
            linewidth=2,
        )
        plt.title("All datasets (avg NDCG@10) — Oracle comparison")
        plt.xlabel("Comparison Budget")
        plt.ylabel("Avg NDCG@10")
        plt.xlim(0, 750)
        plt.tight_layout()
        out_all_oracle = OUTPUT_DIR / "limit_comparisons_oracles_all.png"
        plt.savefig(out_all_oracle)
        print(f"Saved {out_all_oracle}")
        plt.close()

if __name__ == "__main__":
    main()
