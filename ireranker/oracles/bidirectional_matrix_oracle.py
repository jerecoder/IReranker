from __future__ import annotations

from ireranker.config import logger

from .oracle import MatrixKey, MatrixOracle


class BidirectionalMatrixOracle(MatrixOracle):
    """Oracle backed by rerank matrices that store (qid, doc_a, doc_b) and its reverse."""

    def __init__(
        self,
        base_dir=None,
        comparison_limit: int | None = None,
        comparison_limit_per_task: bool = False,
    ):
        super().__init__(
            base_dir=base_dir,
            cache_comparisons=True,
            comparison_limit=comparison_limit,
            comparison_limit_per_task=comparison_limit_per_task,
        )
        self._missing_logs = 0

    def _log_missing_entries(self, forward_key: MatrixKey, reverse_key: MatrixKey) -> None:
        if self._missing_logs < 5:
            logger.warning(f"Missing entries for: {forward_key} {reverse_key}")
            if self._missing_logs == 4:
                logger.warning("Suppressing further missing-entry warnings.")
        self._missing_logs += 1

    def sample_lt(self, i: int, j: int) -> bool:
        forward_pref, reverse_pref = self._pair_preferences(
            i, j, on_missing=self._log_missing_entries
        )
        if forward_pref is None or reverse_pref is None:
            return False

        return forward_pref == "B" and reverse_pref == "A"
