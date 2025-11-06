from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from ireranker.config import PROJ_ROOT, REPORTS_DIR
from ireranker.data.loaders import _beir_supported_name, load_beir_dataset
from ireranker.evaluation.beir import evaluate_rankers_beir
from ireranker.rankers import get_ranker


def run_from_config(config_path: Optional[Path] = None) -> None:
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
    if not raw_ds:
        raise ValueError("Config is missing 'datasets' list")
    ds_names = [_beir_supported_name(d) for d in raw_ds]
    ds_names = [d for d in ds_names if d]
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
    rs = [get_ranker(r, seed=seed) for r in eff_rankers]

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
            rows = evaluate_rankers_beir(rs, dataset, k_values)

            import pandas as pd

            d_out = eff_out_root / d
            d_out.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(d_out / "summary.csv", index=False)
            logger.success(f"Saved BEIR evaluation summary to {d_out / 'summary.csv'}")
        except Exception as e:
            from ireranker.config import EXTERNAL_DATA_DIR

            zip_path = EXTERNAL_DATA_DIR / "beir" / f"{d}.zip"
            logger.error(f"Failed dataset '{d}'. Zip: {zip_path}. Error: {e}")
            err_dir = eff_out_root / d
            err_dir.mkdir(parents=True, exist_ok=True)
            (err_dir / "ERROR.txt").write_text(f"Zip: {zip_path}\nError: {e}\n")
            failed.append(d)

    if failed:
        logger.warning(f"Completed with failures for datasets: {', '.join(failed)}")


def main() -> None:
    run_from_config()


if __name__ == "__main__":
    main()
