#!/usr/bin/env python3
"""
Generate a results table from BEIR evaluation reports.

Creates a table where:
- Columns are datasets (plus Avg NDCG@10 and Avg Comparisons per task)
- Rows are rankers
- Cells contain NDCG@10 values

Usage:
    python scripts/generate_results_table.py --datasets dl-2019 dl-2020 robust04 \
        --rankers "bubble sort (classic)" "quick sort (classic)" \
        --model xl \
        --output results_table.csv

    # Specify oracle per ranker using "ranker:oracle" format:
    python scripts/generate_results_table.py --datasets dl-2019 dl-2020 \
        --rankers "bubble sort (classic):bidirectional" "mohajer (ir):sampling" \
        --model xl
"""

import argparse
import csv
from pathlib import Path
from typing import Optional


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a results table from BEIR evaluation reports"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="List of dataset names (e.g., dl-2019 dl-2020 robust04)",
    )
    parser.add_argument(
        "--rankers",
        nargs="+",
        required=True,
        help='List of ranker names. Use "ranker:oracle" format to specify oracle per ranker (e.g., "bubble sort (classic):bidirectional" "mohajer (ir):sampling")',
    )
    parser.add_argument(
        "--model",
        choices=["large", "xl"],
        default="xl",
        help="Model size to use (large or xl). Default: xl",
    )
    parser.add_argument(
        "--oracle",
        type=str,
        default=None,
        help="Filter by oracle type (e.g., bidirectional, sampling, mixed). If not specified, uses first match.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path. If not specified, prints to stdout.",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default=None,
        help="Path to reports directory. Default: reports/beir-metrics",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "markdown", "latex"],
        default="csv",
        help="Output format. Default: csv",
    )
    return parser.parse_args()


def get_reports_dir(reports_dir_arg: Optional[str]) -> Path:
    """Get the reports directory path."""
    if reports_dir_arg:
        return Path(reports_dir_arg)
    
    # Try to find it relative to script location
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    default_path = repo_root / "reports" / "beir-metrics"
    
    if default_path.exists():
        return default_path
    
    raise FileNotFoundError(
        f"Reports directory not found at {default_path}. "
        "Please specify --reports-dir."
    )


def load_summary(summary_path: Path) -> list[dict]:
    """Load a summary CSV file and return list of rows as dicts."""
    if not summary_path.exists():
        return []
    
    with open(summary_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def find_ranker_row(
    rows: list[dict], ranker_name: str, oracle_filter: Optional[str] = None
) -> Optional[dict]:
    """Find a ranker row in the summary data, optionally filtered by oracle."""
    ranker_name_lower = ranker_name.lower().strip()
    
    for row in rows:
        row_ranker = row.get("ranker", "").lower().strip()
        if row_ranker == ranker_name_lower:
            if oracle_filter:
                row_oracle = row.get("oracle", "").lower().strip()
                if oracle_filter.lower() in row_oracle:
                    return row
            else:
                return row
    return None


def parse_ranker_spec(ranker_spec: str) -> tuple[str, Optional[str]]:
    """
    Parse a ranker specification that may include oracle.
    
    Format: "ranker_name" or "ranker_name:oracle"
    Returns: (ranker_name, oracle or None)
    """
    if ":" in ranker_spec:
        # Split only on the last colon (in case ranker name contains colons)
        parts = ranker_spec.rsplit(":", 1)
        return parts[0].strip(), parts[1].strip()
    return ranker_spec.strip(), None


def generate_table(
    datasets: list[str],
    ranker_specs: list[str],
    model: str,
    reports_dir: Path,
    global_oracle_filter: Optional[str] = None,
) -> tuple[list[str], list[list[str]]]:
    """
    Generate the results table.
    
    Args:
        datasets: List of dataset names
        ranker_specs: List of ranker specifications (can include oracle with "ranker:oracle" format)
        model: Model size (large or xl)
        reports_dir: Path to reports directory
        global_oracle_filter: Global oracle filter (overridden by per-ranker oracle)
    
    Returns:
        - headers: list of column names
        - rows: list of rows, each row is a list of values
    """
    model_dir = reports_dir / f"flan-t5-{model}"
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Headers: Ranker, [datasets...], Avg NDCG@10, Avg Comparisons
    headers = ["Ranker"] + datasets + ["Avg NDCG@10", "Avg Comparisons/Task"]
    
    table_rows = []
    
    for ranker_spec in ranker_specs:
        ranker_name, ranker_oracle = parse_ranker_spec(ranker_spec)
        # Use per-ranker oracle if specified, otherwise use global filter
        oracle_filter = ranker_oracle if ranker_oracle else global_oracle_filter
        
        # Display name includes oracle if specified
        if ranker_oracle:
            display_name = f"{ranker_name} ({ranker_oracle})"
        else:
            display_name = ranker_name
        
        row_values = [display_name]
        ndcg_values = []
        comp_values = []
        
        for dataset in datasets:
            summary_path = model_dir / dataset / "summary.csv"
            summary_data = load_summary(summary_path)
            
            ranker_row = find_ranker_row(summary_data, ranker_name, oracle_filter)
            
            if ranker_row:
                ndcg = ranker_row.get("NDCG", "")
                comparisons_per_task = ranker_row.get("Comparisons_per_task", "")
                
                if ndcg:
                    try:
                        ndcg_float = float(ndcg)
                        row_values.append(f"{ndcg_float:.4f}")
                        ndcg_values.append(ndcg_float)
                    except ValueError:
                        row_values.append(ndcg)
                else:
                    row_values.append("-")
                
                if comparisons_per_task:
                    try:
                        comp_values.append(float(comparisons_per_task))
                    except ValueError:
                        pass
            else:
                row_values.append("-")
        
        # Calculate averages
        if ndcg_values:
            avg_ndcg = sum(ndcg_values) / len(ndcg_values)
            row_values.append(f"{avg_ndcg:.4f}")
        else:
            row_values.append("-")
        
        if comp_values:
            avg_comp = sum(comp_values) / len(comp_values)
            row_values.append(f"{avg_comp:.2f}")
        else:
            row_values.append("-")
        
        table_rows.append(row_values)
    
    return headers, table_rows


def format_csv(headers: list[str], rows: list[list[str]]) -> str:
    """Format table as CSV."""
    import io
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(headers)
    writer.writerows(rows)
    return output.getvalue()


def format_markdown(headers: list[str], rows: list[list[str]]) -> str:
    """Format table as Markdown."""
    lines = []
    
    # Header
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    
    # Rows
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    
    return "\n".join(lines)


def format_latex(headers: list[str], rows: list[list[str]]) -> str:
    """Format table as LaTeX."""
    lines = []
    
    # Column alignment
    col_align = "l" + "c" * (len(headers) - 1)
    
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{" + col_align + "}")
    lines.append(r"\toprule")
    
    # Header
    lines.append(" & ".join(headers) + r" \\")
    lines.append(r"\midrule")
    
    # Rows
    for row in rows:
        # Escape underscores in LaTeX
        escaped_row = [cell.replace("_", r"\_") for cell in row]
        lines.append(" & ".join(escaped_row) + r" \\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{BEIR Evaluation Results}")
    lines.append(r"\label{tab:beir-results}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


def main():
    args = parse_args()
    
    reports_dir = get_reports_dir(args.reports_dir)
    
    headers, rows = generate_table(
        datasets=args.datasets,
        ranker_specs=args.rankers,
        model=args.model,
        reports_dir=reports_dir,
        global_oracle_filter=args.oracle,
    )
    
    # Format output
    if args.format == "csv":
        output = format_csv(headers, rows)
    elif args.format == "markdown":
        output = format_markdown(headers, rows)
    elif args.format == "latex":
        output = format_latex(headers, rows)
    else:
        output = format_csv(headers, rows)
    
    # Write or print output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Table saved to {output_path}")
    else:
        print(output)


if __name__ == "__main__":
    main()
