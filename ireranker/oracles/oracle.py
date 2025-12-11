from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
from numbers import Real
from pathlib import Path
import pickle
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from ireranker.config import logger
from ireranker.types import RankingTask

_SCORE_WARNINGS = 0
_SCORE_WARNINGS_LIMIT = 20


def _log_score_warning(msg: str) -> None:
    """Log score parsing anomalies with a soft cap to avoid log spam."""
    global _SCORE_WARNINGS
    try:
        if _SCORE_WARNINGS < _SCORE_WARNINGS_LIMIT:
            logger.warning(msg)
        elif _SCORE_WARNINGS == _SCORE_WARNINGS_LIMIT:
            logger.warning("Suppressing further score parsing warnings.")
    except Exception:
        pass
    finally:
        _SCORE_WARNINGS += 1


class BudgetExceeded(Exception):
    """Raised when the oracle's comparison limit is exceeded."""


class Oracle(ABC):
    """Abstract oracle that answers pairwise comparison queries."""

    def __init__(
        self,
        comparison_limit: int | None = None,
        comparison_limit_per_task: bool = False,
    ) -> None:
        self.current_task: Optional[RankingTask] = None
        self.seed: Optional[int] = None
        self.name: str = self.__class__.__name__
        self._comparisons: int = 0
        self._comparison_calls: int = 0
        self._cache_hits: int = 0
        self._cache_enabled: bool = False
        self._comparison_cache: dict[tuple[int, int], bool] = {}
        self._cache_signature: tuple[str, tuple[str, ...]] | None = None
        self.comparison_limit = comparison_limit
        self.comparison_limit_per_task = comparison_limit_per_task
        self._task_comparisons = 0

    @abstractmethod
    def load_dataset(
        self,
        dataset: str,
        *,
        split: str = "test",
        query_ids: Optional[Iterable[str]] = None,
        matrix_model: Optional[str] = None,
    ) -> None:
        """Load comparison data for the given dataset, replacing any previous state."""

    def _task_signature(self, task: RankingTask) -> tuple[str, tuple[str, ...]]:
        """Signature used to invalidate the cache when switching tasks."""
        return (task.query_id, tuple(task.candidate_ids))

    def _clear_cache(self) -> None:
        """Drop cached comparisons and any stored signature."""
        self._comparison_cache.clear()
        self._cache_signature = None

    def set_task(self, task: RankingTask) -> None:
        """Set the current ranking task for comparison queries."""
        if self.current_task is task:
            return
        self.current_task = task
        self._task_comparisons = 0
        sig = self._task_signature(task)
        if self._cache_signature != sig:
            self._clear_cache()
            self._cache_signature = sig

    @abstractmethod
    def sample_lt(self, i: int, j: int) -> bool:
        """Return True when doc at index i should be ranked after doc at index j."""

    def set_seed(self, seed: int | None) -> None:
        """Set a deterministic seed for any stochastic behavior."""
        self.seed = seed

    def enable_cache(self, enabled: bool = True) -> None:
        self._cache_enabled = enabled
        self._clear_cache()

    def reset_comparisons(self) -> None:
        self._comparisons = 0
        self._comparison_calls = 0
        self._cache_hits = 0
        self._clear_cache()

    def lt(self, i: int, j: int) -> bool:
        """Compare two candidate indices using optional caching/counting."""
        if self.current_task is None:
            return False

        if self.comparison_limit is not None:
            current_comparisons = (
                self._task_comparisons if self.comparison_limit_per_task else self._comparisons
            )
            if current_comparisons >= self.comparison_limit:
                raise BudgetExceeded(f"Comparison limit of {self.comparison_limit} exceeded.")

        self._comparison_calls += 1
        sig = self._task_signature(self.current_task)
        if self._cache_signature != sig:
            self._clear_cache()
            self._cache_signature = sig

        if self._cache_enabled:
            key = (i, j)
            cached = key in self._comparison_cache
            if cached:
                self._cache_hits += 1
            else:
                self._comparison_cache[key] = self.sample_lt(i, j)
                self._comparisons += 2
                self._task_comparisons += 2
            return self._comparison_cache[key]

        self._comparisons += 1
        self._task_comparisons += 1
        return self.sample_lt(i, j)

    @property
    def comparisons(self) -> int:
        return self._comparisons

    @property
    def comparison_calls(self) -> int:
        return self._comparison_calls

    @property
    def cache_hits(self) -> int:
        return self._cache_hits


MatrixKey = Tuple[str, str, str]


class MatrixOracle(Oracle):
    """Oracle backed by rerank matrices that store (qid, doc_a, doc_b) and its reverse."""

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        cache_comparisons: bool = True,
        comparison_limit: int | None = None,
        comparison_limit_per_task: bool = False,
    ):
        super().__init__(
            comparison_limit=comparison_limit,
            comparison_limit_per_task=comparison_limit_per_task,
        )
        self._base_dir = Path(base_dir) if base_dir else None
        self._matrix: Optional[Dict[MatrixKey, Mapping[str, Any]]] = None
        self._dataset: Optional[str] = None
        self.enable_cache(cache_comparisons)

    def load_dataset(
        self,
        dataset: str,
        *,
        split: str = "test",
        query_ids: Optional[Iterable[str]] = None,
        matrix_model: Optional[str] = None,
    ) -> None:
        self._clear_cache()
        self.current_task = None
        self._matrix = self._load_matrix(dataset, split, matrix_model=matrix_model)
        if query_ids is not None:
            allowed = {str(qid) for qid in query_ids}
            if allowed:
                matrix_qids = {k[0] for k in self._matrix.keys()}
                if not allowed.issuperset(matrix_qids):
                    self._matrix = {k: v for k, v in self._matrix.items() if k[0] in allowed}
        self._dataset = dataset

    def _ensure_matrix_loaded(self) -> Dict[MatrixKey, Mapping[str, Any]]:
        if self._matrix is None:
            raise RuntimeError("Oracle matrix not loaded. Call load_dataset() first.")
        return self._matrix

    def _entry_preference(self, entry: Mapping[str, Any] | None) -> Optional[str]:
        if entry is None:
            return None
        score_a, score_b = self._extract_scores(entry)
        if score_a is None or score_b is None or score_a == score_b:
            return None
        return "A" if score_a > score_b else "B"

    def _pair_preferences(
        self,
        i: int,
        j: int,
        *,
        on_missing: Callable[[MatrixKey, MatrixKey], None] | None = None,
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Return preferences for (qid, doc_i, doc_j) and its reverse, or (None, None).

        Optional on_missing callback is invoked when either matrix entry is absent.
        """
        if self.current_task is None:
            return None, None

        matrix = self._ensure_matrix_loaded()
        qid = self.current_task.query_id
        doc_a = self.current_task.candidate_ids[i]
        doc_b = self.current_task.candidate_ids[j]

        forward_key = (qid, doc_a, doc_b)
        reverse_key = (qid, doc_b, doc_a)

        forward_entry = matrix.get(forward_key)
        reverse_entry = matrix.get(reverse_key)
        if forward_entry is None or reverse_entry is None:
            if on_missing is not None:
                on_missing(forward_key, reverse_key)
            return None, None

        return self._entry_preference(forward_entry), self._entry_preference(reverse_entry)

    @staticmethod
    def _extract_scores(
        entry: Mapping[str, Any],
    ) -> Tuple[Optional[float], Optional[float]]:
        raw_scores = entry.get("scores")
        pairs: List[Tuple[str, float]] = []

        if isinstance(raw_scores, Mapping):
            pairs = [(str(k), raw_scores[k]) for k in raw_scores]
        elif isinstance(raw_scores, (list, tuple)):
            for item in raw_scores:
                if (
                    isinstance(item, tuple)
                    and len(item) == 2
                    and isinstance(item[0], str)
                    and isinstance(item[1], Real)
                ):
                    pairs.append((item[0], float(item[1])))
                else:
                    _log_score_warning(
                        f"Unexpected score tuple format: {item!r} (type {type(item)})"
                    )
        else:
            _log_score_warning(f"Unexpected 'scores' type in matrix entry: {type(raw_scores)}")

        scores: dict[str, float] = {}
        for label, value in pairs:
            if isinstance(value, Real):
                scores[label.strip().upper()] = float(value)
            else:
                _log_score_warning(
                    f"Non-numeric score for label {label!r}: {value!r} (type {type(value)})"
                )

        score_a = scores.get("A")
        score_b = scores.get("B")
        if score_a is None or score_b is None:
            _log_score_warning(
                f"Missing A/B scores in matrix entry; keys present: {sorted(scores.keys())}"
            )
        return score_a, score_b

    def _load_matrix(
        self,
        dataset: str,
        split: str = "test",
        matrix_model: Optional[str] = None,
    ) -> Dict[MatrixKey, Mapping[str, Any]]:
        dataset_key = dataset.lower().strip()
        split_key = split.lower().strip() if split else ""
        model_key = matrix_model.lower().strip() if matrix_model else None
        base = (self._base_dir or _default_base_dir()).resolve()
        return _load_matrix_cached(dataset_key, split_key, model_key, base)


@lru_cache(maxsize=8)
def _load_matrix_cached(
    dataset_key: str, split_key: str, model_key: Optional[str], base: Path
) -> Dict[MatrixKey, Mapping[str, Any]]:
    if not base.exists():
        raise FileNotFoundError(f"Rerank matrix directory not found: {base}")

    candidates: List[Path] = []
    for path in base.rglob("*.pkl"):
        name = path.name.lower()
        if dataset_key in name and (not split_key or split_key in name):
            if model_key and model_key not in name and model_key not in path.parent.name.lower():
                continue
            candidates.append(path)
    if not candidates and split_key:
        for path in base.rglob("*.pkl"):
            name = path.name.lower()
            if dataset_key in name:
                if (
                    model_key
                    and model_key not in name
                    and model_key not in path.parent.name.lower()
                ):
                    continue
                candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No rerank matrix found for dataset '{dataset_key}' in {base}")
    best = max(candidates, key=lambda p: p.stat().st_mtime)
    with best.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Unexpected rerank matrix format in {best}: {type(obj)}")
    return obj  # type: ignore[return-value]


def clear_matrix_cache() -> None:
    """Drop any cached rerank matrices to free memory."""
    _load_matrix_cached.cache_clear()


def load_matrix(
    dataset: str,
    split: str = "test",
    base_dir: Optional[Path] = None,
    matrix_model: Optional[str] = None,
) -> Dict[MatrixKey, Mapping[str, Any]]:
    """Public helper to load a rerank matrix with caching."""
    base = (base_dir or _default_base_dir()).resolve()
    return _load_matrix_cached(
        dataset.lower().strip(),
        split.lower().strip() if split else "",
        matrix_model.lower().strip() if matrix_model else None,
        base,
    )


def _default_base_dir() -> Path:
    from ireranker.config import EXTERNAL_DATA_DIR

    return EXTERNAL_DATA_DIR / "reranking-matrices"
