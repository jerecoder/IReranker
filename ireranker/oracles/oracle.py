from __future__ import annotations

from abc import ABC, abstractmethod
from numbers import Real
from pathlib import Path
import pickle
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ireranker.types import RankingTask


class Oracle(ABC):
    """Abstract oracle that answers pairwise comparison queries."""

    def __init__(self) -> None:
        self.current_task: Optional[RankingTask] = None
        self.seed: Optional[int] = None

    @abstractmethod
    def load_dataset(self, dataset: str, *, split: str = "test") -> None:
        """Load comparison data for the given dataset, replacing any previous state."""

    def set_task(self, task: RankingTask) -> None:
        """Set the current ranking task for comparison queries."""
        self.current_task = task

    @abstractmethod
    def sample_lt(self, i: int, j: int) -> bool:
        """Return True when doc at index i should be ranked after doc at index j."""

    def set_seed(self, seed: int | None) -> None:
        """Set a deterministic seed for any stochastic behavior."""
        self.seed = seed


MatrixKey = Tuple[str, str, str]


class MatrixOracle(Oracle):
    """Oracle backed by rerank matrices that store (qid, doc_a, doc_b) and its reverse."""

    def __init__(self, base_dir: Optional[Path] = None):
        super().__init__()
        self._base_dir = Path(base_dir) if base_dir else None
        self._matrix: Optional[Dict[MatrixKey, Mapping[str, Any]]] = None
        self._dataset: Optional[str] = None

    def load_dataset(self, dataset: str, *, split: str = "test") -> None:
        self._matrix = self._load_matrix(dataset, split)
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
        scores: dict[str, float] = {}
        for label, value in pairs:
            if isinstance(value, Real):
                scores[label.strip().upper()] = float(value)
        score_a = scores.get("A")
        score_b = scores.get("B")
        return score_a, score_b

    def _load_matrix(
        self,
        dataset: str,
        split: str = "test",
    ) -> Dict[MatrixKey, Mapping[str, Any]]:
        dataset_key = dataset.lower().strip()
        split_key = split.lower().strip() if split else ""
        base = self._base_dir or self._default_base_dir()  # type: ignore
        if not base.exists():
            raise FileNotFoundError(f"Rerank matrix directory not found: {base}")

        candidates: List[Path] = []
        for path in base.rglob("*.pkl"):
            name = path.name.lower()
            if dataset_key in name and (not split_key or split_key in name):
                candidates.append(path)
        if not candidates and split_key:
            for path in base.rglob("*.pkl"):
                if dataset_key in path.name.lower():
                    candidates.append(path)
        if not candidates:
            raise FileNotFoundError(
                f"No rerank matrix found for dataset '{dataset}' in {base}"
            )
        best = max(candidates, key=lambda p: p.stat().st_mtime)
        with best.open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, dict):
            raise ValueError(f"Unexpected rerank matrix format in {best}: {type(obj)}")
        return obj  # type: ignore[return-value]

    def _default_base_dir(self) -> Path:
        from ireranker.config import EXTERNAL_DATA_DIR

        return EXTERNAL_DATA_DIR / "reranking-matrices"
