from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ireranker.config import PROJ_ROOT, REPORTS_DIR, logger
from ireranker.data.loaders import _beir_supported_name, load_beir_dataset
from ireranker.evaluation.beir import evaluate_rankers_beir
from ireranker.oracles import BidirectionalMatrixOracle, SamplingMatrixOracle
from ireranker.rankers import get_ranker

_TABLE_START = "<!-- BEGIN_BEIR_RESULTS -->"
_TABLE_END = "<!-- END_BEIR_RESULTS -->"


def _format_mean(value: float | None, *, precision: int = 4, as_int: bool = False) -> str:
    if value is None or (isinstance(value, float) and (value != value)):  # NaN check
        return "n/a"
    if as_int:
        return str(int(round(value)))
    return f"{value:.{precision}f}"


def _format_sci(value: float | None) -> str:
    if value is None or (isinstance(value, float) and (value != value)):
        return "n/a"
    return f"{value:.3e}"


def _datasets_for_table(requested: Sequence[str], out_root: Path) -> List[str]:
    names = {d.lower(): d for d in requested}
    if out_root.exists():
        for child in out_root.iterdir():
            if child.is_dir():
                names[child.name.lower()] = child.name
    return [names[k] for k in sorted(names.keys())]


def _ndcg10_by_dataset(
    out_root: Path, datasets: Sequence[str]
) -> tuple[Dict[str, Dict[str, float]], List[str]]:
    scores: Dict[str, Dict[str, float]] = {}
    ranker_names: set[str] = set()
    for ds in datasets:
        path = out_root / ds / "summary.csv"
        scores[ds] = {}
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        k_val = int(row.get("k", ""))
                    except Exception:
                        continue
                    if k_val != 10:
                        continue
                    ranker = row.get("ranker")
                    if not ranker:
                        continue
                    try:
                        ndcg_val = float(row.get("NDCG", ""))
                    except (TypeError, ValueError):
                        continue
                    scores[ds][ranker] = ndcg_val
                    ranker_names.add(ranker)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"Could not parse {path}: {e}")
    return scores, sorted(ranker_names)


def _render_ndcg10_grid(out_root: Path, datasets: Sequence[str]) -> str:
    score_map, rankers = _ndcg10_by_dataset(out_root, datasets)
    if not rankers:
        rankers = ["n/a"]
    header = "| Dataset | " + " | ".join(rankers) + " |"
    divider = "| --- | " + " | ".join("---" for _ in rankers) + " |"
    lines = [header, divider]
    for ds in datasets:
        row_scores = score_map.get(ds, {})
        cells = []
        for r in rankers:
            val = row_scores.get(r)
            cells.append(_format_mean(val) if val is not None else "n/a")
        lines.append("| " + ds + " | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _ndcg10_rate_by_ranker(out_root: Path, datasets: Sequence[str]) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for ds in datasets:
        path = out_root / ds / "summary.csv"
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        k_val = int(row.get("k", ""))
                    except Exception:
                        continue
                    if k_val != 10:
                        continue
                    ranker = row.get("ranker")
                    if not ranker:
                        continue
                    try:
                        rate = float(row.get("NDCG_per_comp", ""))
                    except (TypeError, ValueError):
                        continue
                    totals[ranker] = totals.get(ranker, 0.0) + rate
                    counts[ranker] = counts.get(ranker, 0) + 1
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"Could not parse {path}: {e}")
    averages: Dict[str, float] = {}
    for rnk, total in totals.items():
        cnt = counts.get(rnk, 0)
        if cnt > 0:
            averages[rnk] = total / cnt
    return averages


def _render_rate_table(out_root: Path, datasets: Sequence[str]) -> str:
    rates = _ndcg10_rate_by_ranker(out_root, datasets)
    if not rates:
        return "| Ranker | Avg NDCG@10/Comparisons |\n| --- | --- |\n| n/a | n/a |"
    header = "| Ranker | Avg NDCG@10/Comparisons |"
    divider = "| --- | --- |"
    body = ["| " + r + " | " + _format_sci(rate) + " |" for r, rate in sorted(rates.items())]
    return "\n".join([header, divider, *body])


def _update_readme_results_table(out_root: Path, requested: Sequence[str]) -> None:
    readme_path = PROJ_ROOT / "README.md"
    all_datasets = _datasets_for_table(requested, out_root)
    rate_table = _render_rate_table(out_root, all_datasets)
    ndcg10_grid = _render_ndcg10_grid(out_root, all_datasets)
    block = f"{_TABLE_START}\n{rate_table}\n\n{ndcg10_grid}\n{_TABLE_END}"
    if readme_path.exists():
        text = readme_path.read_text(encoding="utf-8")
    else:
        text = ""
    if _TABLE_START in text and _TABLE_END in text:
        prefix, rest = text.split(_TABLE_START, 1)
        _, suffix = rest.split(_TABLE_END, 1)
        new_text = prefix + block + suffix
    else:
        section = (
            "\n## BEIR results\n\n"
            "Tables auto-updated after each BEIR evaluation.\n\n"
            f"{block}\n"
        )
        new_text = text.rstrip() + "\n\n" + section
    readme_path.write_text(new_text, encoding="utf-8")


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

    def _oracle_for_ranker(name: str):
        if name.lower() == "mohajer":
            return SamplingMatrixOracle(seed=seed)
        return BidirectionalMatrixOracle()

    rs = [get_ranker(r, seed=seed, oracle=_oracle_for_ranker(r)) for r in eff_rankers]

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
            rows = evaluate_rankers_beir(rs, dataset, k_values, seed=seed)

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
            fieldnames = [
                "ranker",
                "k",
                "NDCG",
                "MAP",
                "Recall",
                "Precision",
                "Comparisons",
                "NDCG_per_comp",
            ]
            with summary_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow({k: row.get(k) for k in fieldnames})
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

    try:
        _update_readme_results_table(eff_out_root, ds_names)
        logger.info("Updated README with BEIR results table.")
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"Could not update README results table: {e}")

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
