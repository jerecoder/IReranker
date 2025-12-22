#!/usr/bin/env python3
"""Example: Using TransformerOracle to generate and use comparison matrices.

This example demonstrates:
1. Initializing TransformerOracle with a model
2. Loading a BEIR dataset
3. Generating comparisons for ranking tasks
4. Saving the generated matrix
5. Using the saved matrix with evaluation

For this example to work, you need:
- A GPU (or CPU, but slow)
- HuggingFace access to Llama models (or use a different model)
- BEIR dataset downloaded

Quick start:
    # Test with a tiny example (no GPU needed)
    python examples/transformer_oracle_example.py --test

    # Generate with real model (requires GPU)
    python examples/transformer_oracle_example.py --model meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
from pathlib import Path

from ireranker.config import logger
from ireranker.oracles import TransformerOracle
from ireranker.rankers import BubbleRanker
from ireranker.types import RankingTask


def create_test_data():
    """Create minimal test data without requiring BEIR."""
    # Simulate a tiny dataset
    queries = {
        "q1": "What is machine learning?",
        "q2": "How does neural network training work?"
    }

    corpus = {
        "d1": "Machine learning is a subset of artificial intelligence.",
        "d2": "Neural networks are trained using backpropagation.",
        "d3": "Deep learning uses multiple layers of neural networks.",
        "d4": "AI systems can learn from data without being explicitly programmed."
    }

    tasks = [
        RankingTask(
            query_id="q1",
            candidate_ids=["d1", "d2", "d3", "d4"],
            y_true=[1.0, 0.3, 0.5, 0.8]
        ),
        RankingTask(
            query_id="q2",
            candidate_ids=["d1", "d2", "d3", "d4"],
            y_true=[0.2, 1.0, 0.9, 0.1]
        )
    ]

    return queries, corpus, tasks


class MockTransformerOracle(TransformerOracle):
    """Mock oracle for testing without GPU."""

    def __init__(self):
        # Skip parent init to avoid loading model
        from ireranker.oracles.oracle import Oracle
        Oracle.__init__(self)

        self._queries = {}
        self._corpus = {}
        self._matrix = {}
        self._dataset_name = "mock"
        self._comparison_count_since_checkpoint = 0
        self._checkpoint_interval = 1000
        self._checkpoint_dir = None
        self.enable_cache(True)

    def _load_model(self):
        # Don't actually load a model
        logger.info("Using mock model (no GPU required)")

    def _prompt_model(self, query, doc_a, doc_b):
        # Simple heuristic: prefer document with more matching words
        import random
        query_words = set(query.lower().split())
        a_words = set(doc_a.lower().split())
        b_words = set(doc_b.lower().split())

        a_score = len(query_words & a_words) / max(len(query_words), 1)
        b_score = len(query_words & b_words) / max(len(query_words), 1)

        # Add some randomness
        a_score += random.random() * 0.1
        b_score += random.random() * 0.1

        if a_score > b_score:
            return "A", 0.6 + a_score * 0.4, 0.4 - b_score * 0.4
        else:
            return "B", 0.4 - a_score * 0.4, 0.6 + b_score * 0.4

    def load_test_data(self, queries, corpus):
        """Load test data directly."""
        self._queries = queries
        self._corpus = corpus


def run_example(use_real_model=False, model_name=None, dataset="scifact"):
    """Run the TransformerOracle example.

    Args:
        use_real_model: If True, load actual transformer model (requires GPU)
        model_name: Model identifier for HuggingFace
        dataset: BEIR dataset name
    """
    logger.info("="*80)
    logger.info("TransformerOracle Example")
    logger.info("="*80)

    # Initialize oracle
    if use_real_model:
        logger.info(f"Initializing TransformerOracle with {model_name}")
        oracle = TransformerOracle(
            model_name=model_name,
            device="cuda",
            quantization="8bit"  # Use quantization to fit in GPU
        )

        # Load BEIR dataset
        logger.info(f"Loading BEIR dataset: {dataset}")
        oracle.load_dataset(dataset, split="test")

        # Enable checkpoints
        checkpoint_dir = Path("data/checkpoints/example")
        oracle.enable_checkpoints(checkpoint_dir, interval=100)

        # Load a few tasks
        from ireranker.data.loaders import load_beir_dataset
        ranking_dataset = load_beir_dataset(dataset, split="test", max_queries=5)
        tasks = ranking_dataset.tasks

    else:
        logger.info("Using mock oracle (no GPU required)")
        oracle = MockTransformerOracle()

        # Create test data
        queries, corpus, tasks = create_test_data()
        oracle.load_test_data(queries, corpus)

    # Initialize ranker
    ranker = BubbleRanker(oracle=oracle)
    logger.info(f"Using ranker: {ranker.__class__.__name__}")

    # Process tasks
    logger.info(f"\nProcessing {len(tasks)} tasks...")
    for i, task in enumerate(tasks):
        logger.info(f"\nTask {i+1}/{len(tasks)}: Query '{task.query_id}'")

        # Reset comparison counter
        oracle.reset_comparisons()

        # Rank candidates (ranker will set task on oracle)
        ranked_indices = ranker.rank(task)
        ranked_ids = [task.candidate_ids[idx] for idx in ranked_indices]

        # Display results
        logger.info(f"  Candidates: {len(task.candidate_ids)}")
        logger.info(f"  Comparisons: {oracle.comparisons}")
        logger.info(f"  Ranked order: {ranked_ids[:5]}...")

        if task.y_true:
            # Show relevance scores for top candidates
            logger.info("  Top 3 relevance scores:")
            for j, doc_id in enumerate(ranked_ids[:3]):
                idx = task.candidate_ids.index(doc_id)
                rel_score = task.y_true[idx]
                logger.info(f"    {j+1}. {doc_id}: {rel_score}")

    # Show matrix statistics
    logger.info(f"\nMatrix statistics:")
    logger.info(f"  Total entries: {len(oracle._matrix)}")
    logger.info(f"  Total comparisons: {oracle.comparisons}")

    # Save matrix
    output_dir = Path("data/examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"example_matrix.pkl"

    logger.info(f"\nSaving matrix to: {output_path}")
    oracle.save_matrix(output_path)

    logger.info("\n" + "="*80)
    logger.info("Example completed!")
    logger.info("="*80)

    if use_real_model:
        logger.info(f"\nNext steps:")
        logger.info(f"1. Move matrix to: data/external/reranking-matrices/")
        logger.info(f"2. Run evaluation: make beir-eval")
        logger.info(f"3. Check results: reports/beir-metrics/")
    else:
        logger.info(f"\nThis was a mock example. To use real models:")
        logger.info(f"  python examples/transformer_oracle_example.py --model meta-llama/Llama-3.1-8B-Instruct")


def main():
    parser = argparse.ArgumentParser(description="TransformerOracle example")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test mode with mock model (no GPU needed)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model identifier"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="scifact",
        help="BEIR dataset name"
    )

    args = parser.parse_args()

    use_real_model = not args.test

    try:
        run_example(
            use_real_model=use_real_model,
            model_name=args.model,
            dataset=args.dataset
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
