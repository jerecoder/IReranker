from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

INPUT_CSV = Path("reports/limit_comparisons_experiment.csv")
TIME_UNIT = "s"  # "ms" or "s"
OUTPUT_DIR = Path("reports/figures/limit_comparisons_time")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SHOW_PLOTS = False

df = pd.read_csv(INPUT_CSV)

style_map = {
    "Bidirectional": {"linestyle": (0, (1, 2)), "marker": "o"},
    "Sampling": {"linestyle": "-", "marker": "s"},
}
marker_offsets = {"mohajer + bubble": 0.2}

rankers_all = sorted(df["Ranker"].unique())
palette = sns.color_palette("tab10", n_colors=len(rankers_all))
color_map = {ranker: palette[idx] for idx, ranker in enumerate(rankers_all)}


def _prepare_df(
    frame: pd.DataFrame,
    time_col: str,
    *,
    dataset_filter: list[str] | None,
    aggregate: bool,
) -> pd.DataFrame:
    if time_col not in frame.columns:
        raise ValueError(
            f"Missing {time_col}. Run scripts/refine_limit_comparisons.py first."
        )

    local = frame
    if dataset_filter:
        local = local[local["Dataset"].isin(dataset_filter)].copy()

    if aggregate:
        local = (
            local.groupby(["Ranker", "Oracle", "Budget"], as_index=False)[
                ["NDCG@10", time_col]
            ]
            .mean()
            .rename(columns={time_col: "time_ms"})
        )
        local["label"] = local["Ranker"] + " [" + local["Oracle"] + "]"
    else:
        local = local.rename(columns={time_col: "time_ms"})
        local["label"] = (
            local["Ranker"]
            + " ["
            + local["Oracle"]
            + "] | "
            + local["Dataset"].astype(str)
        )

    if TIME_UNIT == "s":
        local["time"] = local["time_ms"] / 1000.0
    else:
        local["time"] = local["time_ms"]
    return local


def _plot(
    frame: pd.DataFrame,
    *,
    title: str,
    x_label: str,
    output_path: Path,
) -> None:
    if frame.empty:
        print(f"Skipping empty plot: {title}")
        return

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6))

    for (ranker, oracle, label), group in (
        frame.groupby(["Ranker", "Oracle", "label"], as_index=False)
    ):
        group = group.sort_values("time").reset_index(drop=True)
        if group.empty:
            continue

        peak_pos = int(group["NDCG@10"].idxmax())
        group_to_peak = group.iloc[: peak_pos + 1]
        peak_row = group.iloc[peak_pos]

        color = color_map.get(ranker, "#000000")
        style = style_map.get(oracle, {"linestyle": "-", "marker": "o"})

        ax.plot(
            group_to_peak["time"],
            group_to_peak["NDCG@10"],
            label=label,
            color=color,
            linestyle=style["linestyle"],
            linewidth=2,
        )

        offset = marker_offsets.get(ranker, 0.0)
        if TIME_UNIT == "ms":
            offset *= 1000.0

        ax.scatter(
            group_to_peak["time"] + offset,
            group_to_peak["NDCG@10"],
            marker=style["marker"],
            s=35,
            color=color,
            edgecolor="white",
            linewidths=0.6,
            zorder=3,
        )
        ax.scatter(
            [peak_row["time"] + offset],
            [peak_row["NDCG@10"]],
            marker="x",
            s=80,
            color=color,
            zorder=4,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel("NDCG@10")
    ax.set_title(title)

    style_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=style_map["Sampling"]["linestyle"],
            label="solid = sampling",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=style_map["Bidirectional"]["linestyle"],
            label="dotted = bidirectional",
        ),
    ]
    color_handles = [
        Line2D([0], [0], color=color_map[r], linestyle="-", label=r) for r in rankers_all
    ]

    legend1 = ax.legend(
        handles=style_handles,
        title="Line Style",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    ax.add_artist(legend1)
    ax.legend(
        handles=color_handles,
        title="Ranker Colors",
        bbox_to_anchor=(1.02, 0.82),
        loc="upper left",
        frameon=True,
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved {output_path}")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


plots = [
    {
        "name": "all_flan_a100",
        "time_col": "estimated_ms_a100_flan_t5_xl",
        "dataset_filter": None,
        "aggregate": True,
        "model_label": "flan-t5-xl",
        "title_suffix": "All datasets",
    },
    {
        "name": "all_qwen_a100",
        "time_col": "estimated_ms_a100_qwen_instruct",
        "dataset_filter": None,
        "aggregate": True,
        "model_label": "qwen-instruct",
        "title_suffix": "All datasets",
    },
    {
        "name": "dl19_flan_a100",
        "time_col": "estimated_ms_a100_flan_t5_xl",
        "dataset_filter": ["dl-2019"],
        "aggregate": True,
        "model_label": "flan-t5-xl",
        "title_suffix": "DL2019",
    },
    {
        "name": "dl19_qwen_a100",
        "time_col": "estimated_ms_a100_qwen_instruct",
        "dataset_filter": ["dl-2019"],
        "aggregate": True,
        "model_label": "qwen-instruct",
        "title_suffix": "DL2019",
    },
    {
        "name": "dl20_flan_a100",
        "time_col": "estimated_ms_a100_flan_t5_xl",
        "dataset_filter": ["dl-2020"],
        "aggregate": True,
        "model_label": "flan-t5-xl",
        "title_suffix": "DL2020",
    },
    {
        "name": "dl20_qwen_a100",
        "time_col": "estimated_ms_a100_qwen_instruct",
        "dataset_filter": ["dl-2020"],
        "aggregate": True,
        "model_label": "qwen-instruct",
        "title_suffix": "DL2020",
    },
]

for cfg in plots:
    frame = _prepare_df(
        df,
        cfg["time_col"],
        dataset_filter=cfg["dataset_filter"],
        aggregate=cfg["aggregate"],
    )
    unit = "s" if TIME_UNIT == "s" else "ms"
    x_label = f"Estimated time per task ({unit}) [A100, {cfg['model_label']}]"
    title = (
        "Limit Comparisons: NDCG@10 vs time "
        f"({cfg['title_suffix']}, {cfg['model_label']})"
    )
    out_path = OUTPUT_DIR / f"limit_comparisons_time_{cfg['name']}.png"
    _plot(frame, title=title, x_label=x_label, output_path=out_path)
