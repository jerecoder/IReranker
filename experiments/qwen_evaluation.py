#!/usr/bin/env python3
"""Evaluate all registered rankers using Qwen model comparisons.

This script evaluates rankers using pre-computed Qwen comparisons from pickle files,
measuring NDCG@10, comparisons, and average prompt tokens.

Usage:
    python experiments/qwen_evaluation.py
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, Optional, List
from collections import defaultdict

from beir.retrieval.evaluation import EvaluateRetrieval

from ireranker.data.loaders import load_beir_dataset
from ireranker.rankers import get_ranker, list_rankers
from ireranker.oracles import Oracle, BudgetExceeded
from ireranker.types import RankingTask


class QwenPickleOracle(Oracle):
    """Oracle that loads comparisons from Qwen pickle files."""

    def __init__(self, pickle_path: Path, bidirectional: bool = True,
                 comparison_limit: int | None = None,
                 comparison_limit_per_task: bool = False):
        super().__init__(comparison_limit=comparison_limit,
                        comparison_limit_per_task=comparison_limit_per_task)
        self.pickle_path = pickle_path
        self.bidirectional = bidirectional
        self._matrix = None
        self._prompt_tokens = {}
        self.name = f"Qwen ({'Bidirectional' if bidirectional else 'Sampling'})"
        # Enable caching for bidirectional mode so we count 2 comparisons
        # per unique pair (matching MatrixOracle semantics).
        # Sampling mode keeps caching disabled (single-call behavior).
        self.enable_cache(bool(self.bidirectional))

    def load_dataset(self, dataset: str, *, split: str = "test",
                    query_ids: Optional[List[str]] = None,
                    matrix_model: Optional[str] = None) -> None:
        """Load the pickle file."""
        with open(self.pickle_path, 'rb') as f:
            self._matrix = pickle.load(f)

        # Extract prompt tokens for each comparison
        self._prompt_tokens = {}
        for (qid, doc_a, doc_b), entry in self._matrix.items():
            key = (qid, doc_a, doc_b)
            self._prompt_tokens[key] = entry.get('ptks', 0)

    def sample_lt(self, i: int, j: int) -> bool:
        """Return True if doc i < doc j (j is better)."""
        if self.current_task is None or self._matrix is None:
            return False

        qid = self.current_task.query_id
        doc_a = self.current_task.candidate_ids[i]
        doc_b = self.current_task.candidate_ids[j]

        if self.bidirectional:
            # Bidirectional: both (A,B) and (B,A) must agree
            forward_key = (qid, doc_a, doc_b)
            reverse_key = (qid, doc_b, doc_a)

            forward_entry = self._matrix.get(forward_key)
            reverse_entry = self._matrix.get(reverse_key)

            if forward_entry is None or reverse_entry is None:
                return False

            forward_result = forward_entry['text']  # e.g., "Passage B" or "Passage A"
            reverse_result = reverse_entry['text']

            # i < j means j is better, so we want forward="Passage B" and reverse="Passage A"
            return forward_result == "Passage B" and reverse_result == "Passage A"
        else:
            # Sampling: just use forward comparison
            forward_key = (qid, doc_a, doc_b)
            forward_entry = self._matrix.get(forward_key)

            if forward_entry is None:
                return False

            forward_result = forward_entry['text']
            # i < j means j is better, so we want "Passage B"
            return forward_result == "Passage B"

    def get_prompt_tokens(self, i: int, j: int) -> int:
        """Get prompt tokens for a specific comparison."""
        if self.current_task is None:
            return 0

        qid = self.current_task.query_id
        doc_a = self.current_task.candidate_ids[i]
        doc_b = self.current_task.candidate_ids[j]

        key = (qid, doc_a, doc_b)
        return self._prompt_tokens.get(key, 0)


def evaluate_single_query(ranker, task: RankingTask, oracle: QwenPickleOracle, k: int = 10) -> Dict[str, Any]:
    """Evaluate a ranker on a single query and return metrics."""
    # Track which comparisons were made to calculate avg prompt tokens
    # Record comparisons as (qid, doc_a_id, doc_b_id) to avoid issues
    # when rankers mutate `task.candidate_ids` in-place during ranking.
    comparisons_made: List[tuple] = []
    original_sample_lt = oracle.sample_lt

    def tracked_sample_lt(i: int, j: int) -> bool:
        # Capture current task and doc ids at comparison time
        if oracle.current_task is not None:
            qid = oracle.current_task.query_id
            try:
                doc_a = oracle.current_task.candidate_ids[i]
                doc_b = oracle.current_task.candidate_ids[j]
            except Exception:
                # If indices are out of range for some reason, skip recording
                doc_a = None
                doc_b = None
            comparisons_made.append((qid, doc_a, doc_b))
        else:
            comparisons_made.append((None, None, None))

        return original_sample_lt(i, j)

    # Temporarily replace sample_lt to track comparisons
    oracle.sample_lt = tracked_sample_lt

    # Reset oracle stats for this query
    oracle.reset_comparisons()
    oracle.set_task(task)

    # Rank the task
    ranking = ranker.rank(task)

    # Restore original method
    oracle.sample_lt = original_sample_lt

    # Calculate NDCG@k using BEIR evaluation
    # Convert to BEIR format
    n = len(ranking)
    results = {}
    for pos, idx in enumerate(ranking):
        doc_id = task.candidate_ids[idx]
        score = float(n - pos)
        results[doc_id] = score

    # Create qrels for this query
    qrels = {}
    if task.y_true is not None:
        for doc_id, rel in zip(task.candidate_ids, task.y_true):
            if rel and rel > 0:
                qrels[doc_id] = int(rel)

    # Evaluate using BEIR
    qrels_dict = {task.query_id: qrels}
    results_dict = {task.query_id: results}

    ndcg, _, _, _ = EvaluateRetrieval.evaluate(qrels_dict, results_dict, [k])
    ndcg_at_k = ndcg.get(f"NDCG@{k}", 0.0)

    # Calculate average prompt tokens for comparisons actually made
    if comparisons_made:
        # comparisons_made contains (qid, doc_a, doc_b) tuples now
        prompt_tokens = []
        for qid, doc_a, doc_b in comparisons_made:
            if qid is None or doc_a is None or doc_b is None:
                prompt_tokens.append(0)
                continue
            # Look up prompt tokens directly from the oracle's prompt token map
            prompt_tokens.append(oracle._prompt_tokens.get((qid, doc_a, doc_b), 0))

        avg_prompt_tokens = float(np.mean(prompt_tokens)) if prompt_tokens else 0.0
    else:
        avg_prompt_tokens = 0.0

    return {
        'ndcg@10': ndcg_at_k,
        'comparisons': oracle.comparisons,
        'avg_prompt_tokens': avg_prompt_tokens
    }


def load_bm25_baseline(dataset_name: str) -> Dict[str, float]:
    """Load BM25 baseline NDCG@10 scores from BM25 run files."""
    bm25_run_path = Path(f"data/external/beir/bm25-runs/run.beir.bm25-flat.{dataset_name}.txt")

    if not bm25_run_path.exists():
        print(f"Warning: BM25 run file not found: {bm25_run_path}")
        return {}

    # Load BM25 rankings (TREC format: qid Q0 docid rank score system)
    bm25_rankings = defaultdict(dict)
    with open(bm25_run_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                qid, _, docid, rank, score = parts[0], parts[1], parts[2], parts[3], parts[4]
                bm25_rankings[qid][docid] = float(score)

    # Load qrels for evaluation
    qrels_path = Path(f"data/external/beir/datasets/msmarco/qrels/test.tsv")
    qrels = {}
    if qrels_path.exists():
        with open(qrels_path) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3 and parts[0] != 'query-id':
                    qid, docid, rel = parts[0], parts[1], int(parts[2])
                    if qid not in qrels:
                        qrels[qid] = {}
                    qrels[qid][docid] = rel

    # Calculate NDCG@10 for each query
    bm25_ndcg = {}
    for qid in bm25_rankings:
        if qid not in qrels:
            continue

        # Convert to BEIR format
        results_dict = {qid: bm25_rankings[qid]}
        qrels_dict = {qid: qrels[qid]}

        ndcg, _, _, _ = EvaluateRetrieval.evaluate(qrels_dict, results_dict, [10])
        bm25_ndcg[qid] = ndcg.get("NDCG@10", 0.0)

    return bm25_ndcg


def run_experiment():
    # Qwen models and their pickle files
    QWEN_MODELS = {
        "qwen3-0.6b": {
            "dl-2019": Path("data/external/qwen/qwen3-0.6b_dl19-passage.pkl"),
            "dl-2020": Path("data/external/qwen/qwen3-0.6b_dl20-passage.pkl"),
        },
        "qwen3-4b-instruct": {
            "dl-2019": Path("data/external/qwen/qwen3-4b-instruct-2507_dl19-passage.pkl"),
            "dl-2020": Path("data/external/qwen/qwen3-4b-instruct-2507_dl20-passage.pkl"),
        },
    }

    SEED = 42
    K_VALUES = [10]

    # Output path
    output_path = Path("reports/qwen_evaluation.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results if any
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"Loading existing results from {output_path}")
        try:
            df = pd.read_csv(output_path)
            results = df.to_dict('records')
        except pd.errors.EmptyDataError:
            print("Existing file is empty, starting fresh")
            results = []
    else:
        results = []

    # Helper to check if result exists
    def result_exists(model, dataset, ranker, oracle_type, query_id):
        for r in results:
            if (r["Model"] == model and
                r["Dataset"] == dataset and
                r["Ranker"] == ranker and
                r["Oracle"] == oracle_type and
                r["QueryID"] == query_id):
                return True
        return False

    # Get all registered rankers
    ranker_names = list_rankers()

    print(f"Found {len(ranker_names)} registered rankers:")
    for name in ranker_names:
        print(f"  - {name}")
    print()

    # Count total experiments
    total_experiments = 0
    for model_name, pkl_files in QWEN_MODELS.items():
        for dataset_name, pkl_path in pkl_files.items():
            # Load dataset to get query count
            dataset = load_beir_dataset(dataset_name, split="test")

            for ranker_name in ranker_names:
                for oracle_type in ['bidirectional', 'sampling']:
                    for task in dataset.tasks:
                        if not result_exists(model_name, dataset_name, ranker_name,
                                           oracle_type, task.query_id):
                            total_experiments += 1

    print(f"Total experiments to run: {total_experiments}")

    if total_experiments == 0:
        print("All experiments already completed!")
        return

    # Main evaluation loop
    pbar = tqdm(total=total_experiments, desc="Running experiments")

    for model_name, pkl_files in QWEN_MODELS.items():
        for dataset_name, pkl_path in pkl_files.items():
            print(f"\n{'='*80}")
            print(f"Model: {model_name} | Dataset: {dataset_name}")
            print(f"{'='*80}")

            # Load BM25 baseline scores
            print(f"Loading BM25 baseline for {dataset_name}...")
            bm25_scores = load_bm25_baseline(dataset_name)

            # Add BM25 baseline results for both oracle types
            for oracle_type in ['bidirectional', 'sampling']:
                for qid, ndcg_score in bm25_scores.items():
                    if not result_exists(model_name, dataset_name, 'bm25', oracle_type, qid):
                        result = {
                            'Model': model_name,
                            'Dataset': dataset_name,
                            'Ranker': 'bm25',
                            'Oracle': oracle_type,
                            'QueryID': qid,
                            'NDCG@10': ndcg_score,
                            'Comparisons': 0,
                            'AvgPromptTokens': 0.0
                        }
                        results.append(result)

            # Save BM25 results
            if results:
                df = pd.DataFrame(results)
                df.to_csv(output_path, index=False)
                summary = df.groupby(['Model', 'Dataset', 'Ranker', 'Oracle']).agg({
                    'NDCG@10': 'mean',
                    'Comparisons': 'mean',
                    'AvgPromptTokens': 'mean'
                }).reset_index()
                summary.columns = ['Model', 'Dataset', 'Ranker', 'Oracle',
                                 'Avg_NDCG@10', 'Avg_Comparisons', 'Avg_PromptTokens']
                summary_path = Path("reports/qwen_evaluation_summary.csv")
                summary.to_csv(summary_path, index=False)

            # Load BEIR dataset (candidates from Qwen pickle)
            dataset = load_beir_dataset(dataset_name, split="test")

            # Load BM25 rankings to get proper initial order
            bm25_run_path = Path(f"data/external/beir/bm25-runs/run.beir.bm25-flat.{dataset_name}.txt")
            bm25_rankings = defaultdict(dict)
            if bm25_run_path.exists():
                with open(bm25_run_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            qid, _, docid, rank, score = parts[0], parts[1], parts[2], parts[3], parts[4]
                            bm25_rankings[qid][docid] = float(score)

            # Reorder task candidates by BM25 score
            for task in dataset.tasks:
                if task.query_id in bm25_rankings:
                    bm25_scores_for_query = bm25_rankings[task.query_id]

                    # Sort candidates by BM25 score (descending)
                    sorted_candidates = sorted(
                        task.candidate_ids,
                        key=lambda doc_id: bm25_scores_for_query.get(doc_id, -float('inf')),
                        reverse=True
                    )

                    # Reorder y_true to match
                    if task.y_true is not None:
                        old_to_new_idx = {doc_id: i for i, doc_id in enumerate(sorted_candidates)}
                        new_y_true = [0.0] * len(sorted_candidates)
                        for old_idx, doc_id in enumerate(task.candidate_ids):
                            new_idx = old_to_new_idx[doc_id]
                            new_y_true[new_idx] = task.y_true[old_idx]
                        task.y_true = new_y_true

                    # Update candidate_ids with BM25-ranked order
                    task.candidate_ids = sorted_candidates

            for ranker_name in ranker_names:
                # Skip BM25 - already handled above
                if ranker_name.lower() == 'bm25':
                    continue
                for oracle_type in ['bidirectional', 'sampling']:
                    # Create oracle
                    oracle = QwenPickleOracle(
                        pkl_path,
                        bidirectional=(oracle_type == 'bidirectional')
                    )
                    oracle.load_dataset(dataset_name, split="test")
                    oracle.set_seed(SEED)

                    # Create ranker
                    try:
                        ranker = get_ranker(ranker_name, oracle=oracle, seed=SEED)
                    except Exception as e:
                        print(f"\nSkipping {ranker_name} with {oracle_type}: {e}")
                        continue

                    # Evaluate each query
                    for task in dataset.tasks:
                        # Check cache
                        if result_exists(model_name, dataset_name, ranker_name,
                                       oracle_type, task.query_id):
                            continue

                        pbar.set_description(
                            f"{model_name[:15]} | {dataset_name[:8]} | "
                            f"{ranker_name[:20]} | {oracle_type[:6]} | Q{task.query_id}"
                        )

                        # Evaluate
                        try:
                            metrics = evaluate_single_query(ranker, task, oracle, k=10)

                            result = {
                                'Model': model_name,
                                'Dataset': dataset_name,
                                'Ranker': ranker_name,
                                'Oracle': oracle_type,
                                'QueryID': task.query_id,
                                'NDCG@10': metrics['ndcg@10'],
                                'Comparisons': metrics['comparisons'],
                                'AvgPromptTokens': metrics['avg_prompt_tokens']
                            }

                            results.append(result)

                            # Save after each query
                            df = pd.DataFrame(results)
                            df.to_csv(output_path, index=False)

                            # Generate and save summary
                            summary = df.groupby(['Model', 'Dataset', 'Ranker', 'Oracle']).agg({
                                'NDCG@10': 'mean',
                                'Comparisons': 'mean',
                                'AvgPromptTokens': 'mean'
                            }).reset_index()
                            summary.columns = ['Model', 'Dataset', 'Ranker', 'Oracle',
                                             'Avg_NDCG@10', 'Avg_Comparisons', 'Avg_PromptTokens']
                            summary_path = Path("reports/qwen_evaluation_summary.csv")
                            summary.to_csv(summary_path, index=False)

                        except Exception as e:
                            print(f"\nError: {e}")
                            print(f"  Model: {model_name}, Dataset: {dataset_name}")
                            print(f"  Ranker: {ranker_name}, Oracle: {oracle_type}")
                            print(f"  Query: {task.query_id}")

                        pbar.update(1)

    pbar.close()

    # Generate summary statistics
    print("\n" + "="*80)
    print("GENERATING SUMMARY")
    print("="*80)

    df = pd.DataFrame(results)

    # Calculate averages per model/dataset/ranker/oracle
    summary = df.groupby(['Model', 'Dataset', 'Ranker', 'Oracle']).agg({
        'NDCG@10': 'mean',
        'Comparisons': 'mean',
        'AvgPromptTokens': 'mean'
    }).reset_index()

    summary.columns = ['Model', 'Dataset', 'Ranker', 'Oracle',
                       'Avg_NDCG@10', 'Avg_Comparisons', 'Avg_PromptTokens']

    summary_path = Path("reports/qwen_evaluation_summary.csv")
    summary.to_csv(summary_path, index=False)

    print(f"\nResults saved to: {output_path}")
    print(f"Summary saved to: {summary_path}")

    # Print summary tables
    for model_name in QWEN_MODELS.keys():
        model_data = summary[summary['Model'] == model_name].copy()

        if model_data.empty:
            continue

        print(f"\n{'='*80}")
        print(f"Model: {model_name}")
        print(f"{'='*80}\n")

        # Create pivot table for each dataset
        for dataset in model_data['Dataset'].unique():
            dataset_data = model_data[model_data['Dataset'] == dataset]

            print(f"\n### {dataset}\n")

            # Separate bidirectional and sampling
            bi_data = dataset_data[dataset_data['Oracle'] == 'bidirectional']
            samp_data = dataset_data[dataset_data['Oracle'] == 'sampling']

            print("#### Bidirectional Oracle\n")
            if not bi_data.empty:
                bi_table = bi_data[['Ranker', 'Avg_NDCG@10', 'Avg_Comparisons', 'Avg_PromptTokens']]
                bi_table = bi_table.sort_values('Avg_NDCG@10', ascending=False)
                print(bi_table.to_string(index=False))

            print("\n#### Sampling Oracle\n")
            if not samp_data.empty:
                samp_table = samp_data[['Ranker', 'Avg_NDCG@10', 'Avg_Comparisons', 'Avg_PromptTokens']]
                samp_table = samp_table.sort_values('Avg_NDCG@10', ascending=False)
                print(samp_table.to_string(index=False))

            print()


if __name__ == "__main__":
    run_experiment()
