import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize limit comparisons experiment results.")
    parser.add_argument(
        "--agg-datasets",
        type=str,
        default=None,
        help=(
            "Comma-separated dataset names to use for aggregated plots only "
            "(defaults to all datasets)."
        ),
    )
    args = parser.parse_args()
    agg_datasets = None
    if args.agg_datasets:
        agg_datasets = [d.strip() for d in args.agg_datasets.split(",") if d.strip()]

    # Config
    DATA_PATH = Path("reports/limit_comparisons_experiment.csv")
    OUTPUT_DIR = Path("reports/figures")

    # Create subdirectories
    (OUTPUT_DIR / "main").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "oracles_al").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "oracles_classic").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "aggregated").mkdir(parents=True, exist_ok=True)

    # Set plot style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")

    # Set global font sizes and style
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'font.family': 'sans-serif',
        'axes.grid': True,
        'grid.alpha': 0.6,
        'grid.linewidth': 0.8,
        'axes.facecolor': '#f8f9fa',
        'figure.facecolor': 'white'
    })

    # Define colors - more vibrant and distinct palette
    colors_map = {
        'mohajer (ir)': '#2ecc71',           # Emerald green
        'mohajer + bubble': '#16a085',       # Dark turquoise
        'pac + bubble': '#3498db',           # Bright blue
        'bubble sort (classic)': '#e74c3c',  # Red
        'quick sort (classic)': '#f39c12',   # Orange
        'prp sort (classic)': '#9b59b6',     # Purple
        'jingle bells': '#1abc9c',           # Turquoise
        'christmas tree': '#e67e22',         # Carrot
        'bm25': '#34495e',                   # Dark grey-blue
        'BM25': '#34495e',
    }

    # Verify file exists
    if not DATA_PATH.exists():
        print(f"Error: {DATA_PATH} not found.")
        return

    # Load Data
    df = pd.read_csv(DATA_PATH)

    # Filter Budget <= 700 (exclude 800)
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
        ("mohajer (ir)", "Sampling"),
        ("mohajer + bubble", "Sampling"),
        ("pac + bubble", "Sampling"),
        ("bubble sort (classic)", "Bidirectional"),
        ("quick sort (classic)", "Bidirectional"),
        ("prp sort (classic)", "Bidirectional"),
        ("jingle bells", "Sampling"),
        ("christmas tree", "Sampling"),
        ("bm25", "Bidirectional"),
    ]

    # Set 2: Oracle Comparison, split to reduce clutter
    # AL = Active Learners
    al_rankers = [
        "mohajer (ir)",
        "mohajer + bubble",
        "pac + bubble",
        "jingle bells",
        "christmas tree",
    ]
    classic_rankers = [
        "bubble sort (classic)",
        "quick sort (classic)",
        "prp sort (classic)",
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
            fig, ax = plt.subplots(figsize=(12, 7))

            for ranker in data1["Ranker"].unique():
                ranker_data = data1[data1["Ranker"] == ranker]
                ax.plot(
                    ranker_data["Budget"],
                    ranker_data["NDCG@10"],
                    marker='o',
                    markersize=8,
                    linewidth=2.5,
                    label=ranker,
                    color=colors_map.get(ranker, '#000000'),
                    alpha=0.9,
                    markeredgewidth=1.5,
                    markeredgecolor='white'
                )

            ax.set_xlabel("Inference Budget")
            ax.set_ylabel("NDCG@10")
            ax.set_xlim(0, 750)
            ax.grid(True, alpha=0.6, linestyle='-', linewidth=0.8)
            ax.legend(frameon=True, shadow=True, fancybox=True, loc='best')

            plt.tight_layout()
            out1 = OUTPUT_DIR / "main" / f"limit_comparisons_main_{ds}.png"
            plt.savefig(out1, dpi=300, bbox_inches='tight')
            print(f"Saved {out1}")
            plt.close()

    # ---------------------------------------------------------
    # Plot 2: Oracle Comparison (Active Learning rankers only)
    # ---------------------------------------------------------
        data2_al = ds_data[ds_data["Ranker"].isin(al_rankers)].sort_values("Budget")

        if not data2_al.empty:
            fig, ax = plt.subplots(figsize=(12, 7))

            for ranker in data2_al["Ranker"].unique():
                for oracle in data2_al["OracleDisplay"].unique():
                    subset = data2_al[(data2_al["Ranker"] == ranker) & (data2_al["OracleDisplay"] == oracle)]
                    if subset.empty:
                        continue

                    linestyle = '--' if oracle == "Estocastico" else '-'
                    marker = 's' if oracle == "Estocastico" else 'o'

                    ax.plot(
                        subset["Budget"],
                        subset["NDCG@10"],
                        marker=marker,
                        markersize=7,
                        linewidth=2.5,
                        linestyle=linestyle,
                        label=f"{ranker} [{oracle}]",
                        color=colors_map.get(ranker, '#000000'),
                        alpha=0.9,
                        markeredgewidth=1.5,
                        markeredgecolor='white'
                    )

            ax.set_xlabel("Inference Budget")
            ax.set_ylabel("NDCG@10")
            ax.set_xlim(0, 750)
            ax.grid(True, alpha=0.6, linestyle='-', linewidth=0.8)
            ax.legend(frameon=True, shadow=True, fancybox=True, loc='best', ncol=2)

            plt.tight_layout()
            out2_al = OUTPUT_DIR / "oracles_al" / f"limit_comparisons_oracles_al_{ds}.png"
            plt.savefig(out2_al, dpi=300, bbox_inches='tight')
            print(f"Saved {out2_al}")
            plt.close()

        # ---------------------------------------------------------
        # Plot 3: Oracle Comparison (Classic rankers only)
        # ---------------------------------------------------------
        data2_classic = ds_data[ds_data["Ranker"].isin(classic_rankers)].sort_values("Budget")

        if not data2_classic.empty:
            fig, ax = plt.subplots(figsize=(12, 7))

            for ranker in data2_classic["Ranker"].unique():
                for oracle in data2_classic["OracleDisplay"].unique():
                    subset = data2_classic[(data2_classic["Ranker"] == ranker) & (data2_classic["OracleDisplay"] == oracle)]
                    if subset.empty:
                        continue

                    linestyle = '--' if oracle == "Estocastico" else '-'
                    marker = 's' if oracle == "Estocastico" else 'o'

                    ax.plot(
                        subset["Budget"],
                        subset["NDCG@10"],
                        marker=marker,
                        markersize=7,
                        linewidth=2.5,
                        linestyle=linestyle,
                        label=f"{ranker} [{oracle}]",
                        color=colors_map.get(ranker, '#000000'),
                        alpha=0.9,
                        markeredgewidth=1.5,
                        markeredgecolor='white'
                    )

            ax.set_xlabel("Inference Budget")
            ax.set_ylabel("NDCG@10")
            ax.set_xlim(0, 750)
            ax.grid(True, alpha=0.6, linestyle='-', linewidth=0.8)
            ax.legend(frameon=True, shadow=True, fancybox=True, loc='best', ncol=2)

            plt.tight_layout()
            out2_classic = OUTPUT_DIR / "oracles_classic" / f"limit_comparisons_oracles_classic_{ds}.png"
            plt.savefig(out2_classic, dpi=300, bbox_inches='tight')
            print(f"Saved {out2_classic}")
            plt.close()

    # ---------------------------------------------------------
    # Aggregated plots across datasets (average NDCG@10)
    # ---------------------------------------------------------
    agg_df = df
    if agg_datasets:
        agg_df = df[df["Dataset"].isin(agg_datasets)].copy()
        if agg_df.empty:
            print(
                "No rows found for aggregated datasets: "
                + ", ".join(agg_datasets)
                + ". Skipping aggregated plots."
            )
            agg_df = pd.DataFrame(columns=df.columns)

    agg_main = agg_df[_mask_for_pairs(agg_df, main_comparison_pairs)]
    if not agg_main.empty:
        grouped = (
            agg_main.groupby(["Ranker", "Oracle", "OracleDisplay", "Budget"], as_index=False)[
                "NDCG@10"
            ]
            .mean()
            .sort_values("Budget")
        )

        fig, ax = plt.subplots(figsize=(12, 7))

        for ranker in grouped["Ranker"].unique():
            ranker_data = grouped[grouped["Ranker"] == ranker]
            ax.plot(
                ranker_data["Budget"],
                ranker_data["NDCG@10"],
                marker='o',
                markersize=8,
                linewidth=2.5,
                label=ranker,
                color=colors_map.get(ranker, '#000000'),
                alpha=0.9,
                markeredgewidth=1.5,
                markeredgecolor='white'
            )

        ax.set_xlabel("Inference Budget")
        ax.set_ylabel("Average NDCG@10")
        ax.set_xlim(0, 750)
        ax.grid(True, alpha=0.6, linestyle='-', linewidth=0.8)
        ax.legend(frameon=True, shadow=True, fancybox=True, loc='best')

        plt.tight_layout()
        out_all_main = OUTPUT_DIR / "aggregated" / "limit_comparisons_main_all.png"
        plt.savefig(out_all_main, dpi=300, bbox_inches='tight')
        print(f"Saved {out_all_main}")
        plt.close()

    # Aggregated oracle comparisons (split AL vs Classic)
    agg_oracle_al = agg_df[agg_df["Ranker"].isin(al_rankers)]
    if not agg_oracle_al.empty:
        grouped = (
            agg_oracle_al.groupby(["Ranker", "Oracle", "OracleDisplay", "Budget"], as_index=False)[
                "NDCG@10"
            ]
            .mean()
            .sort_values("Budget")
        )

        fig, ax = plt.subplots(figsize=(12, 7))

        for ranker in grouped["Ranker"].unique():
            for oracle in grouped["OracleDisplay"].unique():
                subset = grouped[(grouped["Ranker"] == ranker) & (grouped["OracleDisplay"] == oracle)]
                if subset.empty:
                    continue

                linestyle = '--' if oracle == "Estocastico" else '-'
                marker = 's' if oracle == "Estocastico" else 'o'

                ax.plot(
                    subset["Budget"],
                    subset["NDCG@10"],
                    marker=marker,
                    markersize=7,
                    linewidth=2.5,
                    linestyle=linestyle,
                    label=f"{ranker} [{oracle}]",
                    color=colors_map.get(ranker, '#000000'),
                    alpha=0.9,
                    markeredgewidth=1.5,
                    markeredgecolor='white'
                )

        ax.set_xlabel("Inference Budget")
        ax.set_ylabel("Average NDCG@10")
        ax.set_xlim(0, 750)
        ax.grid(True, alpha=0.6, linestyle='-', linewidth=0.8)
        ax.legend(frameon=True, shadow=True, fancybox=True, loc='best', ncol=2)

        plt.tight_layout()
        out_all_oracle_al = OUTPUT_DIR / "aggregated" / "limit_comparisons_oracles_all_al.png"
        plt.savefig(out_all_oracle_al, dpi=300, bbox_inches='tight')
        print(f"Saved {out_all_oracle_al}")
        plt.close()

    agg_oracle_classic = agg_df[agg_df["Ranker"].isin(classic_rankers)]
    if not agg_oracle_classic.empty:
        grouped = (
            agg_oracle_classic.groupby(
                ["Ranker", "Oracle", "OracleDisplay", "Budget"], as_index=False
            )["NDCG@10"]
            .mean()
            .sort_values("Budget")
        )

        fig, ax = plt.subplots(figsize=(12, 7))

        for ranker in grouped["Ranker"].unique():
            for oracle in grouped["OracleDisplay"].unique():
                subset = grouped[(grouped["Ranker"] == ranker) & (grouped["OracleDisplay"] == oracle)]
                if subset.empty:
                    continue

                linestyle = '--' if oracle == "Estocastico" else '-'
                marker = 's' if oracle == "Estocastico" else 'o'

                ax.plot(
                    subset["Budget"],
                    subset["NDCG@10"],
                    marker=marker,
                    markersize=7,
                    linewidth=2.5,
                    linestyle=linestyle,
                    label=f"{ranker} [{oracle}]",
                    color=colors_map.get(ranker, '#000000'),
                    alpha=0.9,
                    markeredgewidth=1.5,
                    markeredgecolor='white'
                )

        ax.set_xlabel("Inference Budget")
        ax.set_ylabel("Average NDCG@10")
        ax.set_xlim(0, 750)
        ax.grid(True, alpha=0.6, linestyle='-', linewidth=0.8)
        ax.legend(frameon=True, shadow=True, fancybox=True, loc='best', ncol=2)

        plt.tight_layout()
        out_all_oracle_classic = OUTPUT_DIR / "aggregated" / "limit_comparisons_oracles_all_classic.png"
        plt.savefig(out_all_oracle_classic, dpi=300, bbox_inches='tight')
        print(f"Saved {out_all_oracle_classic}")
        plt.close()

if __name__ == "__main__":
    main()
