from __future__ import annotations

import argparse
import cProfile
import csv
import io
import json
from pathlib import Path
import pstats
import re
from typing import Any, Dict, List, NamedTuple, Optional, Sequence

from ireranker.config import EXTERNAL_DATA_DIR, PROJ_ROOT, REPORTS_DIR, logger
from ireranker.data.loaders import _beir_supported_name, load_beir_dataset
from ireranker.evaluation.beir import evaluate_rankers_beir
from ireranker.oracles.oracle import clear_matrix_cache
from ireranker.rankers import build_rankers_for_eval

_TABLE_START = "<!-- BEGIN_BEIR_RESULTS -->"
_TABLE_END = "<!-- END_BEIR_RESULTS -->"


class _ModelContext(NamedTuple):
    label: str
    model_key: str
    out_root: Path


class _ModelEvalResult(NamedTuple):
    failures: List[str]
    skipped: List[str]
    notes: List[str]
    missing_pair_datasets: set[str]


def _model_slug(model: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")
    return slug or "model"


def _extract_model_name(path: Path) -> Optional[str]:
    """Best-effort model extraction from matrix filename."""
    stem = path.name.rsplit(".", 1)[0]
    parts = stem.split("_")
    if len(parts) < 2:
        return None
    model = parts[-2].strip()
    return model or None


def _discover_matrix_models(base_dir: Path) -> List[str]:
    """Discover available matrix models by scanning the reranking-matrices directory."""
    models: set[str] = set()
    if not base_dir.exists():
        return []
    for path in base_dir.rglob("*.pkl"):
        model = _extract_model_name(path)
        if model:
            models.add(model)
    return sorted(models)


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


def _bold(text: str) -> str:
    return f"**{text}**"


def _ranker_label(row: Dict[str, object]) -> str:
    base = str(row.get("ranker") or "").strip()
    oracle = str(row.get("oracle") or "").strip()
    if oracle:
        return f"{base} [{oracle}]"
    return base


def _datasets_for_table(requested: Sequence[str], out_root: Path) -> List[str]:
    names = {d.lower(): d for d in requested}
    if out_root.exists():
        for child in out_root.iterdir():
            if child.is_dir():
                names[child.name.lower()] = child.name
    return [names[k] for k in sorted(names.keys())]


def _ndcg10_by_dataset(
    out_root: Path, datasets: Sequence[str]
) -> tuple[Dict[str, Dict[str, Dict[str, float | int]]], List[str]]:
    scores: Dict[str, Dict[str, Dict[str, float | int]]] = {}
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
                    ranker = _ranker_label(row)
                    if not ranker:
                        continue
                    try:
                        ndcg_val = float(row.get("NDCG", ""))
                    except (TypeError, ValueError):
                        continue
                    try:
                        comps_val = int(row.get("Comparisons", 0))
                    except (TypeError, ValueError):
                        comps_val = 0
                    try:
                        hits_val = int(row.get("CacheHits", 0))
                    except (TypeError, ValueError):
                        hits_val = 0
                    scores[ds][ranker] = {
                        "ndcg": ndcg_val,
                        "comparisons": comps_val,
                        "cache_hits": hits_val,
                    }
                    ranker_names.add(ranker)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"Could not parse {path}: {e}")
    return scores, sorted(ranker_names)


def _render_ndcg10_grid(
    out_root: Path,
    datasets: Sequence[str],
    *,
    flagged_missing: set[str] | None = None,
    rate_map: Dict[str, Dict[str, float]] | None = None,
) -> str:
    score_map, rankers = _ndcg10_by_dataset(out_root, datasets)
    if not rankers:
        rankers = ["n/a"]
    flagged_missing = flagged_missing or set()
    rate_map = rate_map or {}

    best_per_ds: Dict[str, float | None] = {}
    for ds in datasets:
        ds_scores = score_map.get(ds, {})
        best_val = None
        for v in ds_scores.values():
            if v and v.get("ndcg") is not None:
                ndcg_val = v.get("ndcg")
                if best_val is None or ndcg_val > best_val:
                    best_val = ndcg_val
        best_per_ds[ds] = best_val

    ranker_avgs: Dict[str, float | None] = {}
    for r in rankers:
        vals: List[float] = []
        for ds in datasets:
            if ds in flagged_missing:
                continue
            entry = score_map.get(ds, {}).get(r)
            ndcg_val = entry.get("ndcg") if isinstance(entry, dict) else None
            if ndcg_val is not None:
                vals.append(ndcg_val)
        ranker_avgs[r] = sum(vals) / len(vals) if vals else None

    best_avg = None
    for val in ranker_avgs.values():
        if val is not None and (best_avg is None or val > best_avg):
            best_avg = val

    best_comp = None
    best_hits = None
    if rate_map:
        comps_vals = [
            v.get("comparisons") for v in rate_map.values() if v.get("comparisons") is not None
        ]
        hits_vals = [
            v.get("cache_hits") for v in rate_map.values() if v.get("cache_hits") is not None
        ]
        if comps_vals:
            best_comp = min(comps_vals)
        if hits_vals:
            best_hits = min(hits_vals)

    headers = ["Ranker"]
    for ds in datasets:
        is_flagged = ds in flagged_missing
        label = f"{ds} (missing comparisons)" if is_flagged else ds
        if is_flagged:
            label = f'<span style="color:#c62828">{label}</span>'
        headers.append(label)
    headers.extend(["Average", "Avg #Inference", "Avg Cache Hits"])

    divider = "| " + " | ".join("---" for _ in headers) + " |"
    lines = ["| " + " | ".join(headers) + " |", divider]

    def _sort_key(ranker: str) -> tuple[int, float, str]:
        avg = ranker_avgs.get(ranker)
        if avg is None:
            return (1, 0.0, ranker.lower())
        return (0, -avg, ranker.lower())

    for r in sorted(rankers, key=_sort_key):
        row_cells = [r]
        for ds in datasets:
            ds_scores = score_map.get(ds, {})
            entry = ds_scores.get(r)
            ndcg_val = entry.get("ndcg") if isinstance(entry, dict) else None
            ds_best = best_per_ds.get(ds)
            if ndcg_val is not None:
                ndcg_cell = _format_mean(ndcg_val)
                if ds_best is not None and ndcg_val == ds_best:
                    ndcg_cell = _bold(ndcg_cell)
            else:
                ndcg_cell = "n/a"
            row_cells.append(ndcg_cell)

        avg_val = ranker_avgs.get(r)
        if avg_val is not None:
            avg_cell = _format_mean(avg_val)
            if best_avg is not None and avg_val == best_avg:
                avg_cell = _bold(avg_cell)
        else:
            avg_cell = "n/a"
        row_cells.append(avg_cell)

        comp_val = None if r not in rate_map else rate_map[r].get("comparisons")
        hits_val = None if r not in rate_map else rate_map[r].get("cache_hits")

        if comp_val is not None:
            comp_cell = _format_mean(comp_val, as_int=True)
            if best_comp is not None and comp_val == best_comp:
                comp_cell = _bold(comp_cell)
        else:
            comp_cell = "n/a"
        if hits_val is not None:
            hits_cell = _format_mean(hits_val, as_int=True)
            if best_hits is not None and hits_val == best_hits:
                hits_cell = _bold(hits_cell)
        else:
            hits_cell = "n/a"

        row_cells.extend([comp_cell, hits_cell])

        lines.append("| " + " | ".join(row_cells) + " |")
    return "\n".join(lines)


def _avg_counts_by_ranker(out_root: Path, datasets: Sequence[str]) -> Dict[str, Dict[str, float]]:
    totals: Dict[str, int] = {}
    hit_totals: Dict[str, int] = {}
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
                    ranker = _ranker_label(row)
                    if not ranker:
                        continue
                    try:
                        comps = int(row.get("Comparisons", 0))
                    except (TypeError, ValueError):
                        comps = 0
                    try:
                        hits = int(row.get("CacheHits", 0))
                    except (TypeError, ValueError):
                        hits = 0
                    totals[ranker] = totals.get(ranker, 0) + comps
                    hit_totals[ranker] = hit_totals.get(ranker, 0) + hits
                    counts[ranker] = counts.get(ranker, 0) + 1
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"Could not parse {path}: {e}")
    averages: Dict[str, Dict[str, float]] = {}
    for rnk, total in totals.items():
        cnt = counts.get(rnk, 0)
        if cnt > 0:
            averages[rnk] = {
                "comparisons": total / cnt,
                "cache_hits": hit_totals.get(rnk, 0) / cnt,
            }
    return averages


def _render_rate_table(out_root: Path, datasets: Sequence[str]) -> str:
    avgs = _avg_counts_by_ranker(out_root, datasets)
    if not avgs:
        return "| Ranker | Avg Calls | Avg Cache Hits |\n| --- | --- | --- |\n| n/a | n/a | n/a |"
    header = "| Ranker | Avg Calls | Avg Cache Hits |"
    divider = "| --- | --- | --- |"
    best_comp = min(v["comparisons"] for v in avgs.values()) if avgs else None
    best_hits = min(v["cache_hits"] for v in avgs.values()) if avgs else None
    body = []
    for r, vals in sorted(avgs.items()):
        comp_val = _format_mean(vals["comparisons"], as_int=True)
        hits_val = _format_mean(vals["cache_hits"], as_int=True)
        if best_comp is not None and vals["comparisons"] == best_comp:
            comp_val = _bold(comp_val)
        if best_hits is not None and vals["cache_hits"] == best_hits:
            hits_val = _bold(hits_val)
        body.append("| " + r + " | " + comp_val + " | " + hits_val + " |")
    return "\n".join([header, divider, *body])


def _update_readme_results_table(
    models: Sequence[_ModelContext],
    requested: Sequence[str],
    *,
    notes: Optional[Dict[str, List[str]]] = None,
    flagged: Optional[Dict[str, set[str]]] = None,
) -> None:
    readme_path = PROJ_ROOT / "README.md"
    blocks: list[str] = []
    for ctx in models:
        all_datasets = _datasets_for_table(requested, ctx.out_root)
        flagged_ds = (flagged or {}).get(ctx.label, set())
        rate_map = _avg_counts_by_ranker(ctx.out_root, all_datasets)
        ndcg10_grid = _render_ndcg10_grid(
            ctx.out_root, all_datasets, flagged_missing=flagged_ds, rate_map=rate_map
        )
        heading = f"### {ctx.label}\n" if ctx.label else ""
        block = f"{heading}{ndcg10_grid}"
        model_notes = (notes or {}).get(ctx.label) if notes else None
        if model_notes:
            note_lines = "\n".join(f"- {n}" for n in model_notes)
            block = f"{block}\n\nNotes:\n{note_lines}"
        blocks.append(block)
    block = f"{_TABLE_START}\n" + "\n\n".join(blocks) + f"\n{_TABLE_END}"
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
    rankers_override: Optional[List[str]] = None,
    max_queries_override: Optional[int] = None,
    profile_output: Optional[Path] = None,
    profile_limit: int = 30,
    profile_sort: str = "cumulative",
    skip_readme_update: bool = False,
    matrix_models_override: Optional[List[str]] = None,
    oracle_filter: Optional[str] = None,
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
        # Support comma-separated dataset names
        if isinstance(dataset_override, list):
            raw_ds = dataset_override
        else:
            raw_ds = [d.strip() for d in dataset_override.split(",") if d.strip()]
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
    if rankers_override is not None:
        eff_rankers = rankers_override

    raw_models = cfg.get("matrix_models") if isinstance(cfg.get("matrix_models"), list) else None
    matrix_models: List[str] = []
    if raw_models:
        matrix_models = [str(m).strip() for m in raw_models if str(m).strip()]
    if matrix_models_override is not None:
        matrix_models = [str(m).strip() for m in matrix_models_override if str(m).strip()]
    if not matrix_models:
        discovered = _discover_matrix_models(EXTERNAL_DATA_DIR / "reranking-matrices")
        if discovered:
            matrix_models = discovered
    matrix_models = [m for m in matrix_models if m]
    seen_models: set[str] = set()
    deduped: List[str] = []
    for m in matrix_models:
        key = m.lower()
        if key in seen_models:
            continue
        seen_models.add(key)
        deduped.append(m)
    matrix_models = deduped
    if not matrix_models:
        raise ValueError(
            "No rerank matrix models specified or discovered. "
            "Provide matrix_models in config or --matrix-models."
        )

    seed = int(cfg.get("seed") or 123)

    rs = build_rankers_for_eval(eff_rankers, seed=seed)
    if oracle_filter:
        before_count = len(rs)
        # Allow comma-separated filters (OR logic)
        filters = [f.strip().lower() for f in oracle_filter.split(",") if f.strip()]
        rs = [r for r in rs if any(f in (r.oracle_label or "").lower() for f in filters)]
        logger.info(f"Filtered rankers by oracle '{oracle_filter}': kept {len(rs)}/{before_count}")
    cfg_kvals = cfg.get("k_values") if isinstance(cfg.get("k_values"), list) else None
    k_values = list(cfg_kvals) if cfg_kvals else [1, 3, 5, 10, 100]

    cfg_out = cfg.get("output_dir")
    if isinstance(cfg_out, str) and cfg_out:
        p = Path(cfg_out)
        eff_out_root = p if p.is_absolute() else (REPORTS_DIR / p)
    else:
        eff_out_root = REPORTS_DIR / "beir-metrics"

    model_contexts: List[_ModelContext] = []
    for m in matrix_models:
        out_root = eff_out_root / _model_slug(m)
        model_contexts.append(_ModelContext(label=m, model_key=m, out_root=out_root))

    split = str(cfg.get("split") or "test")
    max_queries = cfg.get("max_queries")
    max_queries = int(max_queries) if max_queries not in (None, "", "null") else None
    if max_queries_override is not None:
        max_queries = max_queries_override
    if profile_output and max_queries is None:
        logger.warning(
            "Profiling without max_queries override; consider a small limit to avoid heavy runs."
        )

    def _run_eval(ctx: _ModelContext) -> _ModelEvalResult:
        failed: List[str] = []
        skipped: List[str] = []
        notes: List[str] = []
        flagged_missing_pairs: set[str] = set()
        logger.info(
            f"Evaluating rerank matrices for model '{ctx.label}' " f"(output: {ctx.out_root})"
        )
        clear_matrix_cache()
        for d in ds_names:
            try:
                logger.info(
                    f"Loading BEIR dataset: {d} (split={split}, matrix_model={ctx.model_key})"
                )
                dataset = load_beir_dataset(
                    d,
                    split=split,
                    max_queries=max_queries,
                    matrix_model=ctx.model_key,
                )
                meta = getattr(dataset, "metadata", {}) or {}
                missing_pairs = int(meta.get("missing_pairs_total") or 0)
                missing_pair_queries = int(meta.get("missing_pairs_queries") or 0)
                if missing_pairs:
                    top_pairs = meta.get("missing_pairs_top") or []
                    top_str = (
                        ", ".join(f"{qid}({cnt})" for qid, cnt in top_pairs) if top_pairs else ""
                    )
                    notes.append(
                        f"{d}: missing {missing_pairs} pair comparisons across {missing_pair_queries} queries"
                        + (f" (top: {top_str})" if top_str else "")
                    )
                    flagged_missing_pairs.add(d)
                task_qids = [t.query_id for t in dataset.tasks]
                for ranker in rs:
                    ranker.set_dataset(
                        d,
                        split=split,
                        query_ids=task_qids,
                        matrix_model=ctx.model_key,
                    )
                rows = evaluate_rankers_beir(rs, dataset, k_values, seed=seed)

                d_out = ctx.out_root / d
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
                    "oracle",
                    "k",
                    "NDCG",
                    "MAP",
                    "Recall",
                    "Precision",
                    "Comparisons",
                    "Comparisons_per_task",
                    "CacheHits",
                    "NDCG_per_comp",
                ]
                with summary_path.open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in rows:
                        writer.writerow({k: row.get(k) for k in fieldnames})
                logger.success(f"Saved BEIR evaluation summary to {d_out / 'summary.csv'}")  # type: ignore
            except FileNotFoundError as e:
                from ireranker.config import EXTERNAL_DATA_DIR

                zip_path = EXTERNAL_DATA_DIR / "beir" / f"{d}.zip"
                logger.warning(
                    f"Skipping dataset '{d}' for model '{ctx.label}' (missing rerank matrix). "
                    f"Zip: {zip_path}. Error: {e}"
                )
                notes.append(f"{d}: skipped (missing rerank matrix)")
                err_dir = ctx.out_root / d
                err_dir.mkdir(parents=True, exist_ok=True)
                summary_path = err_dir / "summary.csv"
                error_path = err_dir / "ERROR.txt"
                skip_path = err_dir / "SKIPPED.txt"
                for path in (summary_path, error_path, skip_path):
                    if path.exists():
                        try:
                            path.unlink()
                        except OSError:
                            logger.warning(f"Could not remove stale file {path}")
                skip_path.write_text(f"Zip: {zip_path}\nReason: {e}\n", encoding="utf-8")
                skipped.append(d)
            except Exception as e:
                from ireranker.config import EXTERNAL_DATA_DIR

                zip_path = EXTERNAL_DATA_DIR / "beir" / f"{d}.zip"
                logger.error(
                    f"Failed dataset '{d}' for model '{ctx.label}'. Zip: {zip_path}. Error: {e}"
                )
                err_dir = ctx.out_root / d
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
            finally:
                clear_matrix_cache()
        return _ModelEvalResult(
            failures=failed,
            skipped=skipped,
            notes=notes,
            missing_pair_datasets=flagged_missing_pairs,
        )

    def _run_all_models() -> Dict[str, _ModelEvalResult]:
        results: Dict[str, _ModelEvalResult] = {}
        for ctx in model_contexts:
            results[ctx.label] = _run_eval(ctx)
        return results

    results: Dict[str, _ModelEvalResult] = {}
    if profile_output:
        prof = cProfile.Profile()
        prof.enable()
        try:
            results = _run_all_models()
        finally:
            prof.disable()
            profile_output.parent.mkdir(parents=True, exist_ok=True)
            prof.dump_stats(profile_output)
            try:
                stats = pstats.Stats(prof).strip_dirs().sort_stats(profile_sort)
                stream = io.StringIO()
                stats.stream = stream  # type: ignore
                stats.print_stats(profile_limit)
                logger.info(
                    "Top %d profile rows (sort=%s):\n%s",
                    profile_limit,
                    profile_sort,
                    stream.getvalue(),
                )
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(f"Could not render profile summary: {e}")
            logger.info(f"Profile stats saved to {profile_output}")
    else:
        results = _run_all_models()

    if not skip_readme_update:
        try:
            notes_by_model = {label: res.notes for label, res in results.items()}
            flagged_by_model = {label: res.missing_pair_datasets for label, res in results.items()}
            _update_readme_results_table(
                model_contexts, ds_names, notes=notes_by_model, flagged=flagged_by_model
            )
            logger.info("Updated README with BEIR results table.")
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"Could not update README results table: {e}")
    else:
        logger.info("Skip README update requested; results left on disk.")

    failed_summary = {label: res.failures for label, res in results.items() if res.failures}
    if failed_summary:
        for label, ds in failed_summary.items():
            logger.warning(
                f"Completed with failures for model '{label}' datasets: {', '.join(ds)}"
            )
    skipped_summary = {label: res.skipped for label, res in results.items() if res.skipped}
    if skipped_summary:
        for label, ds in skipped_summary.items():
            logger.info(
                f"Skipped datasets for model '{label}' (missing rerank matrix): {', '.join(ds)}"
            )


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
        help="Run evaluation only on these dataset names (comma-separated, e.g. trec-covid,scifact).",
    )
    parser.add_argument(
        "--light",
        action="store_true",
        help="Skip datasets listed in config/light_exclude.",
    )
    parser.add_argument(
        "--rankers",
        type=str,
        default=None,
        help="Comma-separated ranker names to run (overrides config).",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Limit number of queries per dataset for a quick/profiled run.",
    )
    parser.add_argument(
        "--profile-out",
        type=str,
        default=None,
        help="Optional path to write cProfile stats for the whole run.",
    )
    parser.add_argument(
        "--profile-limit",
        type=int,
        default=30,
        help="Number of rows to show in the console profile summary (when --profile-out is set).",
    )
    parser.add_argument(
        "--profile-sort",
        type=str,
        default="cumulative",
        help="Sort key for the console profile summary (cumulative, time, calls, etc.).",
    )
    parser.add_argument(
        "--skip-readme-update",
        action="store_true",
        help="Do not rewrite README tables (useful for dry-run/profiling).",
    )
    parser.add_argument(
        "--matrix-models",
        type=str,
        default=None,
        help=(
            "Comma-separated rerank matrix models (matches rerank filename/path). "
            "Examples: flan-t5-large,flan-t5-xl. "
            "Required unless provided in config; falls back to auto-discovery."
        ),
    )
    parser.add_argument(
        "--oracle",
        type=str,
        default=None,
        help="Filter to run only rankers with this oracle label (case-insensitive substring match).",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config) if args.config else None
    rankers_override = None
    if args.rankers:
        rankers_override = [r.strip() for r in args.rankers.split(",") if r.strip()]
    matrix_models_override = None
    if args.matrix_models:
        matrix_models_override = [m.strip() for m in args.matrix_models.split(",") if m.strip()]
    profile_out = Path(args.profile_out) if args.profile_out else None
    run_from_config(
        cfg_path,
        dataset_override=args.dataset,
        light_mode=args.light,
        rankers_override=rankers_override,
        max_queries_override=args.max_queries,
        profile_output=profile_out,
        profile_limit=args.profile_limit,
        profile_sort=args.profile_sort,
        skip_readme_update=args.skip_readme_update,
        matrix_models_override=matrix_models_override,
        oracle_filter=args.oracle,
    )


if __name__ == "__main__":
    main()
