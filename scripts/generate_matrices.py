#!/usr/bin/env python3
"""Script to generate reranking matrices using TransformerOracle.

This script can be run locally or on Google Colab to generate pairwise
comparison matrices for BEIR datasets using transformer models.

Usage:
    # Generate matrices for all configured models and datasets
    python scripts/generate_matrices.py

    # Generate for specific model
    python scripts/generate_matrices.py --model llama-8b

    # Generate for specific datasets
    python scripts/generate_matrices.py --datasets scifact dl-2019

    # Test mode (only 10 queries)
    python scripts/generate_matrices.py --test

    # Resume from checkpoint
    python scripts/generate_matrices.py --resume checkpoints/llama-8b/scifact_checkpoint_5000.pkl
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from ireranker.config import EXTERNAL_DATA_DIR, PROJ_ROOT, logger
from ireranker.oracles import TransformerOracle


def load_config() -> Dict:
    """Load Llama model configuration."""
    config_path = PROJ_ROOT / "config" / "llama_models.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        return json.load(f)


def download_beir_dataset(dataset: str, beir_dir: Path) -> None:
    """Download a BEIR dataset if not already present."""
    dataset_path = beir_dir / dataset
    if dataset_path.exists():
        logger.info(f"Dataset {dataset} already exists at {dataset_path}")
        return

    try:
        from beir import util  # type: ignore
    except ImportError:
        raise ImportError(
            "beir package required for dataset download. "
            "Install with: pip install beir"
        )

    logger.info(f"Downloading dataset: {dataset}")
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    util.download_and_unzip(url, str(beir_dir))
    logger.info(f"Dataset {dataset} downloaded successfully")


def generate_matrix_for_dataset(
    oracle: TransformerOracle,
    dataset: str,
    output_dir: Path,
    max_queries: Optional[int] = None,
) -> Dict:
    """Generate matrix for a single dataset.

    Args:
        oracle: Initialized TransformerOracle
        dataset: BEIR dataset name
        output_dir: Directory to save output matrix
        max_queries: Optional limit on number of queries

    Returns:
        Dictionary with generation statistics
    """
    logger.info(f"Processing dataset: {dataset}")
    start_time = time.time()

    try:
        # Load dataset
        oracle.load_dataset(dataset, split="test")

        # Simulate ranking to trigger comparisons
        # In a real scenario, you'd integrate with a ranker
        # For now, we'll generate comparisons for top-k documents per query

        # Get queries
        queries = list(oracle._queries.items())
        if max_queries:
            queries = queries[:max_queries]

        logger.info(f"Generating comparisons for {len(queries)} queries")

        total_comparisons = 0
        for idx, (qid, query_text) in enumerate(queries):
            # Get candidate documents for this query
            # In practice, you'd get this from retrieval results (e.g., BM25 top-100)
            # For now, we'll skip this and let the rankers trigger comparisons

            if (idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                logger.info(
                    f"Progress: {idx + 1}/{len(queries)} queries, "
                    f"{total_comparisons} comparisons, {elapsed:.1f}s"
                )

        # Save matrix
        output_path = output_dir / f"{dataset}.pkl"
        oracle.save_matrix(output_path)

        elapsed = time.time() - start_time

        result = {
            "dataset": dataset,
            "queries": len(queries),
            "comparisons": oracle.comparisons,
            "matrix_entries": len(oracle._matrix),
            "time_seconds": elapsed,
            "output_path": str(output_path),
            "status": "success"
        }

        logger.info(
            f"Completed {dataset}: {len(queries)} queries, "
            f"{oracle.comparisons} comparisons, {elapsed:.1f}s"
        )

        return result

    except Exception as e:
        logger.error(f"Error processing {dataset}: {e}")
        import traceback
        traceback.print_exc()

        return {
            "dataset": dataset,
            "status": "error",
            "error": str(e)
        }


def main():
    parser = argparse.ArgumentParser(
        description="Generate reranking matrices using transformer models"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model key from config (e.g., 'llama-8b'). If not specified, uses all models."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Dataset names (e.g., 'scifact dl-2019'). If not specified, uses config defaults."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: only process 10 queries per dataset"
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume from checkpoint file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for matrices (default: data/external/reranking-matrices/llama)"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Directory for checkpoint files (default: data/checkpoints)"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1000,
        help="Number of comparisons between checkpoints (default: 1000)"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config()

    # Determine models to use
    if args.model:
        if args.model not in config["models"]:
            raise ValueError(f"Unknown model: {args.model}. Available: {list(config['models'].keys())}")
        models = {args.model: config["models"][args.model]}
    else:
        models = config["models"]

    # Determine datasets to use
    if args.datasets:
        datasets = args.datasets
    else:
        # Default to small + medium datasets
        datasets = config["datasets"]["small"] + config["datasets"]["medium"]

    # Set up directories
    if args.output_dir:
        output_base = args.output_dir
    else:
        output_base = EXTERNAL_DATA_DIR / "reranking-matrices" / "llama"

    if args.checkpoint_dir:
        checkpoint_base = args.checkpoint_dir
    else:
        checkpoint_base = PROJ_ROOT / "data" / "checkpoints"

    # Download datasets
    beir_dir = EXTERNAL_DATA_DIR / "beir"
    for dataset in datasets:
        download_beir_dataset(dataset, beir_dir)

    # Process each model
    all_results = []

    for model_key, model_config in models.items():
        logger.info("=" * 80)
        logger.info(f"Model: {model_key} ({model_config['name']})")
        logger.info("=" * 80)

        # Initialize oracle
        oracle = TransformerOracle(
            model_name=model_config["name"],
            device="cuda",
            quantization=model_config.get("quantization"),
        )

        # Set up checkpoints
        checkpoint_dir = checkpoint_base / model_key
        oracle.enable_checkpoints(checkpoint_dir, interval=args.checkpoint_interval)

        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            oracle.load_matrix(args.resume)

        # Create output directory for this model
        output_dir = output_base / model_key
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process each dataset
        for dataset in datasets:
            max_queries = 10 if args.test else None
            result = generate_matrix_for_dataset(
                oracle, dataset, output_dir, max_queries=max_queries
            )
            result["model"] = model_key
            all_results.append(result)

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("Generation Summary")
    logger.info("=" * 80)

    successful = [r for r in all_results if r["status"] == "success"]
    failed = [r for r in all_results if r["status"] == "error"]

    if successful:
        logger.info(f"\nSuccessful: {len(successful)}")
        total_comparisons = sum(r["comparisons"] for r in successful)
        total_time = sum(r["time_seconds"] for r in successful)
        logger.info(f"  Total comparisons: {total_comparisons:,}")
        logger.info(f"  Total time: {total_time / 3600:.2f} hours")

        for result in successful:
            logger.info(
                f"  ✓ {result['model']}/{result['dataset']}: "
                f"{result['queries']} queries, {result['comparisons']} comparisons, "
                f"{result['time_seconds']:.1f}s"
            )

    if failed:
        logger.info(f"\nFailed: {len(failed)}")
        for result in failed:
            logger.info(f"  ✗ {result['model']}/{result['dataset']}: {result['error']}")

    # Save results summary
    results_file = output_base / "generation_summary.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
