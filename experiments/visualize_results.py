
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

    # Display-only Oracle labels (leave metrics untouched)
    def _oracle_display(row):
        if str(row["Ranker"]).lower() == "bm25":
            return "BM25"
        mapping = {"Sampling": "Estocastico", "Bidirectional": "Bidireccional"}
        return mapping.get(row["Oracle"], row["Oracle"])

    df["OracleDisplay"] = df.apply(_oracle_display, axis=1)
    
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
    
    # Set 2: Oracle Comparison, split to reduce clutter
    ir_rankers = [
        "Mohajer (IR)",
        "PAC + Bubble",
        "Jingle Bells",
        "Christmas Tree",
    ]
    classic_rankers = [
        "Bubble Sort (Classic)",
        "Quick Sort (Classic)",
        "PRP Sort (classic)",
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
    # Plot 2: Oracle Comparison (IR rankers only)
    # ---------------------------------------------------------
        data2_ir = ds_data[ds_data["Ranker"].isin(ir_rankers)].sort_values("Budget")
        
        if not data2_ir.empty:
            plt.figure(figsize=(10, 6))
            sns.lineplot(
                data=data2_ir, 
                x="Budget", 
                y="NDCG@10", 
                hue="Ranker", 
                style="OracleDisplay",
                palette=colors_map, 
                markers=True,
                dashes={"Bidireccional": "", "Estocastico": (2, 2), "BM25": ""},
                linewidth=2
            )
            plt.title(f"{ds}")
            plt.xlabel("Comparison Budget")
            plt.ylabel("NDCG@10")
            plt.xlim(0, 750)
            plt.tight_layout()
            
            out2_ir = OUTPUT_DIR / f"limit_comparisons_oracles_ir_{ds}.png"
            plt.savefig(out2_ir)
            print(f"Saved {out2_ir}")
            plt.close()

        # ---------------------------------------------------------
        # Plot 3: Oracle Comparison (Classic rankers only)
        # ---------------------------------------------------------
        data2_classic = ds_data[ds_data["Ranker"].isin(classic_rankers)].sort_values("Budget")
        
        if not data2_classic.empty:
            plt.figure(figsize=(10, 6))
            sns.lineplot(
                data=data2_classic, 
                x="Budget", 
                y="NDCG@10", 
                hue="Ranker", 
                style="OracleDisplay",
                palette=colors_map, 
                markers=True,
                dashes={"Bidireccional": "", "Estocastico": (2, 2), "BM25": ""},
                linewidth=2
            )
            plt.title(f"{ds}")
            plt.xlabel("Comparison Budget")
            plt.ylabel("NDCG@10")
            plt.xlim(0, 750)
            plt.tight_layout()
            
            out2_classic = OUTPUT_DIR / f"limit_comparisons_oracles_classic_{ds}.png"
            plt.savefig(out2_classic)
            print(f"Saved {out2_classic}")
            plt.close()

    # ---------------------------------------------------------
    # Aggregated plots across all datasets (average NDCG@10)
    # ---------------------------------------------------------
    agg_main = df[_mask_for_pairs(df, main_comparison_pairs)]
    if not agg_main.empty:
        grouped = (
            agg_main.groupby(["Ranker", "Oracle", "OracleDisplay", "Budget"], as_index=False)[
                "NDCG@10"
            ]
            .mean()
            .sort_values("Budget")
        )
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=grouped,
            x="Budget",
            y="NDCG@10",
            hue="Ranker",
            style="OracleDisplay",
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

    # Aggregated oracle comparisons (split IR vs Classic)
    agg_oracle_ir = df[df["Ranker"].isin(ir_rankers)]
    if not agg_oracle_ir.empty:
        grouped = (
            agg_oracle_ir.groupby(["Ranker", "Oracle", "OracleDisplay", "Budget"], as_index=False)[
                "NDCG@10"
            ]
            .mean()
            .sort_values("Budget")
        )
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=grouped,
            x="Budget",
            y="NDCG@10",
            hue="Ranker",
            style="OracleDisplay",
            palette=colors_map,
            markers=True,
            dashes={"Bidireccional": "", "Estocastico": (2, 2), "BM25": ""},
            linewidth=2,
        )
        plt.title("All datasets (avg NDCG@10) — Oracle comparison (IR)")
        plt.xlabel("Comparison Budget")
        plt.ylabel("Avg NDCG@10")
        plt.xlim(0, 750)
        plt.tight_layout()
        out_all_oracle_ir = OUTPUT_DIR / "limit_comparisons_oracles_all_ir.png"
        plt.savefig(out_all_oracle_ir)
        print(f"Saved {out_all_oracle_ir}")
        plt.close()

    agg_oracle_classic = df[df["Ranker"].isin(classic_rankers)]
    if not agg_oracle_classic.empty:
        grouped = (
            agg_oracle_classic.groupby(
                ["Ranker", "Oracle", "OracleDisplay", "Budget"], as_index=False
            )["NDCG@10"]
            .mean()
            .sort_values("Budget")
        )
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=grouped,
            x="Budget",
            y="NDCG@10",
            hue="Ranker",
            style="OracleDisplay",
            palette=colors_map,
            markers=True,
            dashes={"Bidireccional": "", "Estocastico": (2, 2), "BM25": ""},
            linewidth=2,
        )
        plt.title("All datasets (avg NDCG@10) — Oracle comparison (Classic)")
        plt.xlabel("Comparison Budget")
        plt.ylabel("Avg NDCG@10")
        plt.xlim(0, 750)
        plt.tight_layout()
        out_all_oracle_classic = OUTPUT_DIR / "limit_comparisons_oracles_all_classic.png"
        plt.savefig(out_all_oracle_classic)
        print(f"Saved {out_all_oracle_classic}")
        plt.close()

if __name__ == "__main__":
    main()
