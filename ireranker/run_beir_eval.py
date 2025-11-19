from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from ireranker.config import PROJ_ROOT, REPORTS_DIR
from ireranker.data.loaders import _beir_supported_name, load_beir_dataset
from ireranker.evaluation.beir import evaluate_rankers_beir
from ireranker.rankers import get_ranker
from ireranker.types import BidirectionalMatrixOracle


def run_from_config(
    config_path: Optional[Path] = None,
    *,
    dataset_override: Optional[str] = None,
    light_mode: bool = False,
) -> None:
    cfg_path = config_path or (PROJ_ROOT / "config" / "beir_eval.json")

    cfg: Dict[str, Any] = {}
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            logger.info(f"Loaded config from {cfg_path}")
        except Exception as e:
            logger.warning(f"Could not read config {cfg_path}: {e}")
    else:
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    raw_ds = cfg.get("datasets") if isinstance(cfg.get("datasets"), list) else None
    if dataset_override:
        raw_ds = [dataset_override]
    if not raw_ds:
        raise ValueError("Config is missing 'datasets' list")
    ds_names = [_beir_supported_name(d) for d in raw_ds]
    ds_names = [d for d in ds_names if d]
    if light_mode and dataset_override is None:
        cfg_exclude = cfg.get("light_exclude")
        exclude = (
            {_beir_supported_name(d) for d in cfg_exclude}
            if isinstance(cfg_exclude, list)
            else set()
        )
        exclude = {d for d in exclude if d}
        if exclude:
            before = set(ds_names)
            ds_names = [d for d in ds_names if d not in exclude]
            skipped = sorted(before - set(ds_names))
            if skipped:
                logger.info(f"Light mode skipping datasets: {', '.join(skipped)}")
    if not ds_names:
        raise ValueError("No supported BEIR datasets to evaluate.")

    cfg_rankers = cfg.get("rankers")
    if isinstance(cfg_rankers, str):
        cfg_rankers = [cfg_rankers]
    if not cfg_rankers or cfg_rankers == ["all"]:
        from ireranker.rankers import list_rankers as _list

        eff_rankers = _list()
    else:
        eff_rankers = cfg_rankers

    seed = int(cfg.get("seed") or 123)
    rs = [get_ranker(r, seed=seed, oracle=BidirectionalMatrixOracle()) for r in eff_rankers]

    cfg_kvals = cfg.get("k_values") if isinstance(cfg.get("k_values"), list) else None
    k_values = list(cfg_kvals) if cfg_kvals else [1, 3, 5, 10, 100]

    cfg_out = cfg.get("output_dir")
    if isinstance(cfg_out, str) and cfg_out:
        p = Path(cfg_out)
        eff_out_root = p if p.is_absolute() else (REPORTS_DIR / p)
    else:
        eff_out_root = REPORTS_DIR / "beir-metrics"

    split = str(cfg.get("split") or "test")
    max_queries = cfg.get("max_queries")
    max_queries = int(max_queries) if max_queries not in (None, "", "null") else None

    failed: List[str] = []
    for d in ds_names:
        try:
            logger.info(f"Loading BEIR dataset: {d} (split={split})")
            dataset = load_beir_dataset(
                d,
                split=split,
                max_queries=max_queries,
            )
            for ranker in rs:
                ranker.set_dataset(d, split=split)
            rows = evaluate_rankers_beir(rs, dataset, k_values)

            import pandas as pd

            d_out = eff_out_root / d
            d_out.mkdir(parents=True, exist_ok=True)
            summary_path = d_out / "summary.csv"
            error_path = d_out / "ERROR.txt"
            for old in (summary_path, error_path):
                if old.exists():
                    try:
                        old.unlink()
                    except OSError:
                        logger.warning(f"Could not remove stale file {old}")
            pd.DataFrame(rows).to_csv(summary_path, index=False)
            logger.success(f"Saved BEIR evaluation summary to {d_out / 'summary.csv'}")  # type: ignore
        except Exception as e:
            from ireranker.config import EXTERNAL_DATA_DIR

            zip_path = EXTERNAL_DATA_DIR / "beir" / f"{d}.zip"
            logger.error(f"Failed dataset '{d}'. Zip: {zip_path}. Error: {e}")
            err_dir = eff_out_root / d
            err_dir.mkdir(parents=True, exist_ok=True)
            summary_path = err_dir / "summary.csv"
            if summary_path.exists():
                try:
                    summary_path.unlink()
                except OSError:
                    logger.warning(f"Could not remove stale file {summary_path}")
            err_file = err_dir / "ERROR.txt"
            if err_file.exists():
                try:
                    err_file.unlink()
                except OSError:
                    logger.warning(f"Could not remove stale file {err_file}")
            err_file.write_text(f"Zip: {zip_path}\nError: {e}\n")
            failed.append(d)

    if failed:
        logger.warning(f"Completed with failures for datasets: {', '.join(failed)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate rankers on BEIR datasets.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a custom beir_eval.json file.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Run evaluation only on this dataset name.",
    )
    parser.add_argument(
        "--light",
        action="store_true",
        help="Skip datasets listed in config/light_exclude.",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config) if args.config else None
    run_from_config(cfg_path, dataset_override=args.dataset, light_mode=args.light)


if __name__ == "__main__":
    main()
