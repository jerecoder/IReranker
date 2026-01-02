#!/usr/bin/env python3
"""Analyze average prompt tokens in TREC reranking matrices.

This script loads the pickle matrices from data/external/reranking-matrices/Own
and calculates average prompt tokens (tokens_in) for DL-2019 and DL-2020.

Usage:
    python experiments/analyze_prompt_tokens.py
"""

from __future__ import annotations

import pickle
import statistics
from pathlib import Path
from typing import Any, Dict, List


# Path to reranking matrices
MATRICES_DIR = Path(__file__).parent.parent / "data" / "external" / "reranking-matrices" / "Own"


def load_matrix(path: Path) -> Dict[Any, Dict[str, Any]]:
    """Load a pickle matrix file."""
    print(f"Loading {path.name}...")
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"  Loaded {len(data)} entries")
    return data


def extract_tokens_in(matrix: Dict[Any, Dict[str, Any]]) -> List[int]:
    """Extract all ptks (prompt tokens) values from a matrix."""
    tokens_list = []
    for key, entry in matrix.items():
        if isinstance(entry, dict) and "ptks" in entry:
            tokens_list.append(entry["ptks"])
    return tokens_list


def compute_stats(tokens_list: List[int], name: str) -> Dict[str, float]:
    """Compute statistics for a list of token counts."""
    if not tokens_list:
        print(f"  {name}: No data")
        return {}
    
    mean = statistics.mean(tokens_list)
    variance = statistics.variance(tokens_list) if len(tokens_list) > 1 else 0
    std = statistics.stdev(tokens_list) if len(tokens_list) > 1 else 0
    min_val = min(tokens_list)
    max_val = max(tokens_list)
    
    print(f"  {name}:")
    print(f"    Count:    {len(tokens_list):,}")
    print(f"    Mean:     {mean:.2f} tokens")
    print(f"    Std Dev:  {std:.2f}")
    print(f"    Variance: {variance:.2f}")
    print(f"    Min:      {min_val}")
    print(f"    Max:      {max_val}")
    
    return {
        "name": name,
        "count": len(tokens_list),
        "mean": mean,
        "std": std,
        "variance": variance,
        "min": min_val,
        "max": max_val,
    }


def main():
    print("=" * 60)
    print("PROMPT TOKENS ANALYSIS")
    print("=" * 60)
    print()
    
    # Find all pickle files
    if not MATRICES_DIR.exists():
        print(f"ERROR: Directory not found: {MATRICES_DIR}")
        return 1
    
    pkl_files = list(MATRICES_DIR.glob("*.pkl"))
    if not pkl_files:
        print(f"ERROR: No pickle files found in {MATRICES_DIR}")
        return 1
    
    print(f"Found {len(pkl_files)} pickle files in {MATRICES_DIR.name}/")
    print()
    
    # Group by dataset
    dl19_tokens: List[int] = []
    dl20_tokens: List[int] = []
    all_tokens: List[int] = []
    
    for pkl_path in sorted(pkl_files):
        matrix = load_matrix(pkl_path)
        tokens = extract_tokens_in(matrix)
        
        if not tokens:
            print(f"  WARNING: No tokens_in found in {pkl_path.name}")
            continue
        
        all_tokens.extend(tokens)
        
        # Categorize by dataset name
        name_lower = pkl_path.name.lower()
        if "dl19" in name_lower or "dl-19" in name_lower or "dl2019" in name_lower:
            dl19_tokens.extend(tokens)
            print(f"  -> Added to DL-2019 ({len(tokens):,} entries)")
        elif "dl20" in name_lower or "dl-20" in name_lower or "dl2020" in name_lower:
            dl20_tokens.extend(tokens)
            print(f"  -> Added to DL-2020 ({len(tokens):,} entries)")
        else:
            print(f"  -> Unknown dataset (added to overall only)")
        print()
    
    # Compute and display statistics
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()
    
    stats_dl19 = compute_stats(dl19_tokens, "DL-2019")
    print()
    stats_dl20 = compute_stats(dl20_tokens, "DL-2020")
    print()
    stats_all = compute_stats(all_tokens, "Overall (All Datasets)")
    print()
    
    # Summary table
    print("=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print()
    print(f"{'Dataset':<25} {'Count':>12} {'Mean':>12} {'Std Dev':>12} {'Variance':>14}")
    print("-" * 75)
    
    for stats in [stats_dl19, stats_dl20, stats_all]:
        if stats:
            print(f"{stats['name']:<25} {stats['count']:>12,} {stats['mean']:>12.2f} {stats['std']:>12.2f} {stats['variance']:>14.2f}")
    
    print()
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
