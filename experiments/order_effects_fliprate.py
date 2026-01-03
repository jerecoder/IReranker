"""
Order Effects Flip-Rate Experiment

Measures positional bias in pairwise LLM judges by computing the flip-rate:
the fraction of pairs where the winner changes when document order is reversed.

Results are stratified by:
- BM25 rank-distance (|r(i) - r(j)| bins)
- Hardness (score margin from the matrix)
"""

import itertools
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from typing import Optional, Tuple, Dict, List

from ireranker.data.loaders import load_beir_dataset
from ireranker.oracles.oracle import MatrixOracle, load_matrix
from ireranker.types import RankingTask


def is_flip(forward_pref: Optional[str], reverse_pref: Optional[str]) -> Optional[bool]:
    """
    Determine if a pair exhibits a flip (order effect).
    
    Agreement (no flip):
        - forward="A" and reverse="B" means doc_i wins consistently
        - forward="B" and reverse="A" means doc_j wins consistently
    
    Flip (order effect):
        - Any other combination where preferences don't agree
    
    Returns:
        True if flip, False if consistent, None if missing data
    """
    if forward_pref is None or reverse_pref is None:
        return None
    # Consistent: (A,B) or (B,A)
    consistent = (forward_pref == "A" and reverse_pref == "B") or \
                 (forward_pref == "B" and reverse_pref == "A")
    return not consistent


def get_rank_distance_bin(i: int, j: int) -> str:
    """Bin pairs by BM25 rank distance."""
    dist = abs(i - j)
    if dist <= 5:
        return "1-5"
    if dist <= 10:
        return "6-10"
    if dist <= 20:
        return "11-20"
    return ">20"


def get_hardness_category(score_a: Optional[float], score_b: Optional[float]) -> str:
    """
    Categorize pair by hardness based on score margin.
    
    Near-tie: small score difference (hard to distinguish)
    Easy: large score difference (clear winner)
    """
    if score_a is None or score_b is None:
        return "unknown"
    margin = abs(score_a - score_b)
    # Typical scores are log-likelihoods or probabilities
    # We'll use relative threshold based on the max
    max_score = max(abs(score_a), abs(score_b), 0.001)
    relative_margin = margin / max_score
    
    if relative_margin < 0.1:
        return "near-tie"
    elif relative_margin < 0.3:
        return "moderate"
    else:
        return "easy"


class FlipRateAnalyzer:
    """Analyzes flip-rates from reranking matrices."""
    
    def __init__(self, matrix_model: str = "flan-t5-xl"):
        self.matrix_model = matrix_model
        self.matrix: Dict = {}
        
    def load_dataset(self, dataset: str, split: str = "test") -> None:
        """Load the reranking matrix for a dataset."""
        self.matrix = load_matrix(dataset, split=split, matrix_model=self.matrix_model)
        self._dataset = dataset
        
    def analyze_pair(
        self, 
        qid: str, 
        doc_i: str, 
        doc_j: str,
        idx_i: int,
        idx_j: int
    ) -> Dict:
        """
        Analyze a single pair for flip-rate computation.
        
        Returns dict with:
            - forward_pref, reverse_pref
            - is_flip
            - rank_distance_bin
            - hardness
            - score margins
        """
        forward_key = (qid, doc_i, doc_j)
        reverse_key = (qid, doc_j, doc_i)
        
        forward_entry = self.matrix.get(forward_key)
        reverse_entry = self.matrix.get(reverse_key)
        
        result = {
            "qid": qid,
            "doc_i": doc_i,
            "doc_j": doc_j,
            "idx_i": idx_i,
            "idx_j": idx_j,
            "rank_distance": abs(idx_i - idx_j),
            "rank_distance_bin": get_rank_distance_bin(idx_i, idx_j),
            "forward_pref": None,
            "reverse_pref": None,
            "is_flip": None,
            "forward_score_a": None,
            "forward_score_b": None,
            "reverse_score_a": None,
            "reverse_score_b": None,
            "hardness": "unknown",
            "missing_data": False,
        }
        
        if forward_entry is None or reverse_entry is None:
            result["missing_data"] = True
            return result
            
        # Extract scores
        forward_scores = self._extract_scores(forward_entry)
        reverse_scores = self._extract_scores(reverse_entry)
        
        result["forward_score_a"] = forward_scores[0]
        result["forward_score_b"] = forward_scores[1]
        result["reverse_score_a"] = reverse_scores[0]
        result["reverse_score_b"] = reverse_scores[1]
        
        # Compute preferences
        forward_pref = self._entry_preference(forward_scores)
        reverse_pref = self._entry_preference(reverse_scores)
        
        result["forward_pref"] = forward_pref
        result["reverse_pref"] = reverse_pref
        result["is_flip"] = is_flip(forward_pref, reverse_pref)
        
        # Compute hardness from forward scores (could also use reverse or average)
        result["hardness"] = get_hardness_category(forward_scores[0], forward_scores[1])
        
        return result
    
    def _extract_scores(self, entry: Dict) -> Tuple[Optional[float], Optional[float]]:
        """Extract A and B scores from a matrix entry."""
        raw_scores = entry.get("scores")
        scores = {}
        
        if isinstance(raw_scores, dict):
            for k, v in raw_scores.items():
                scores[str(k).strip().upper()] = float(v)
        elif isinstance(raw_scores, (list, tuple)):
            for item in raw_scores:
                if isinstance(item, tuple) and len(item) == 2:
                    scores[str(item[0]).strip().upper()] = float(item[1])
                    
        return scores.get("A"), scores.get("B")
    
    def _entry_preference(self, scores: Tuple[Optional[float], Optional[float]]) -> Optional[str]:
        """Determine preference from scores."""
        score_a, score_b = scores
        if score_a is None or score_b is None or score_a == score_b:
            return None
        return "A" if score_a > score_b else "B"


def run_experiment():
    """Run the order effects flip-rate experiment."""
    
    # Configuration
    datasets = ["dl-2019", "dl-2020"]
    matrix_model = "flan-t5-xl"
    max_pairs_per_task = None  # None = all pairs, set to limit for speed
    
    output_path = Path("reports/order_effects_fliprate.csv")
    summary_path = Path("reports/order_effects_fliprate_summary.csv")
    
    all_results = []
    analyzer = FlipRateAnalyzer(matrix_model=matrix_model)
    
    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*60}")
        
        # Load dataset
        dataset = load_beir_dataset(dataset_name, split="test", matrix_model=matrix_model)
        analyzer.load_dataset(dataset_name)
        
        for task in tqdm(dataset.tasks, desc=f"{dataset_name} tasks"):
            qid = task.query_id
            candidates = task.candidate_ids
            n_candidates = len(candidates)
            
            # Generate all pairs
            all_pairs = list(itertools.combinations(range(n_candidates), 2))
            
            # Optionally sample
            if max_pairs_per_task and len(all_pairs) > max_pairs_per_task:
                np.random.seed(42)
                pair_indices = np.random.choice(len(all_pairs), max_pairs_per_task, replace=False)
                pairs = [all_pairs[i] for i in pair_indices]
            else:
                pairs = all_pairs
                
            for idx_i, idx_j in pairs:
                doc_i = candidates[idx_i]
                doc_j = candidates[idx_j]
                
                result = analyzer.analyze_pair(qid, doc_i, doc_j, idx_i, idx_j)
                result["dataset"] = dataset_name
                all_results.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Save raw results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nRaw results saved to {output_path}")
    
    # Compute summary statistics
    print("\n" + "="*60)
    print("FLIP-RATE ANALYSIS RESULTS")
    print("="*60)
    
    # Filter valid pairs (no missing data)
    valid_df = df[~df["missing_data"] & df["is_flip"].notna()]
    
    # Overall flip-rate
    overall_flip_rate = valid_df["is_flip"].mean()
    print(f"\nOverall Flip-Rate: {overall_flip_rate:.2%}")
    print(f"Total valid pairs analyzed: {len(valid_df):,}")
    print(f"Pairs with missing data: {df['missing_data'].sum():,}")
    
    # Flip-rate by dataset
    print("\n--- Flip-Rate by Dataset ---")
    dataset_summary = valid_df.groupby("dataset").agg(
        flip_rate=("is_flip", "mean"),
        n_pairs=("is_flip", "count"),
        n_flips=("is_flip", "sum")
    ).round(4)
    print(dataset_summary.to_string())
    
    # Flip-rate by rank-distance bin
    print("\n--- Flip-Rate by BM25 Rank-Distance ---")
    bin_order = ["1-5", "6-10", "11-20", ">20"]
    rank_summary = valid_df.groupby("rank_distance_bin").agg(
        flip_rate=("is_flip", "mean"),
        n_pairs=("is_flip", "count"),
        n_flips=("is_flip", "sum")
    ).reindex(bin_order).round(4)
    print(rank_summary.to_string())
    
    # Flip-rate by hardness
    print("\n--- Flip-Rate by Hardness ---")
    hardness_summary = valid_df.groupby("hardness").agg(
        flip_rate=("is_flip", "mean"),
        n_pairs=("is_flip", "count"),
        n_flips=("is_flip", "sum")
    ).round(4)
    print(hardness_summary.to_string())
    
    # Cross-tabulation: rank-distance x hardness
    print("\n--- Flip-Rate: Rank-Distance x Hardness ---")
    cross_tab = valid_df.pivot_table(
        index="rank_distance_bin",
        columns="hardness",
        values="is_flip",
        aggfunc="mean"
    ).reindex(bin_order).round(4)
    print(cross_tab.to_string())
    
    # Flip-rate by dataset and rank-distance
    print("\n--- Flip-Rate: Dataset x Rank-Distance ---")
    dataset_rank = valid_df.pivot_table(
        index="dataset",
        columns="rank_distance_bin",
        values="is_flip",
        aggfunc="mean"
    ).reindex(columns=bin_order).round(4)
    print(dataset_rank.to_string())
    
    # Save summary to CSV
    summary_data = []
    
    # Overall
    summary_data.append({
        "category": "overall",
        "subcategory": "all",
        "flip_rate": overall_flip_rate,
        "n_pairs": len(valid_df),
        "n_flips": valid_df["is_flip"].sum()
    })
    
    # By dataset
    for ds, row in dataset_summary.iterrows():
        summary_data.append({
            "category": "dataset",
            "subcategory": ds,
            "flip_rate": row["flip_rate"],
            "n_pairs": row["n_pairs"],
            "n_flips": row["n_flips"]
        })
    
    # By rank-distance
    for rd, row in rank_summary.iterrows():
        if pd.notna(row["flip_rate"]):
            summary_data.append({
                "category": "rank_distance",
                "subcategory": rd,
                "flip_rate": row["flip_rate"],
                "n_pairs": row["n_pairs"],
                "n_flips": row["n_flips"]
            })
    
    # By hardness
    for h, row in hardness_summary.iterrows():
        summary_data.append({
            "category": "hardness",
            "subcategory": h,
            "flip_rate": row["flip_rate"],
            "n_pairs": row["n_pairs"],
            "n_flips": row["n_flips"]
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")
    
    return df, summary_df


if __name__ == "__main__":
    run_experiment()
