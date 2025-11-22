from __future__ import annotations

import random

from ireranker.types import RankingTask

from .oracle import MatrixOracle


class SamplingMatrixOracle(MatrixOracle):
    """Oracle that samples direction when both forward/reverse preferences exist."""

    def sample_lt(self, task: RankingTask, i: int, j: int) -> bool:
        matrix = self._ensure_matrix_loaded()

        qid = task.query_id
        doc_a = task.candidate_ids[i]
        doc_b = task.candidate_ids[j]

        forward_key = (qid, doc_a, doc_b)
        reverse_key = (qid, doc_b, doc_a)

        forward_entry = matrix.get(forward_key)
        reverse_entry = matrix.get(reverse_key)
        if forward_entry is None or reverse_entry is None:
            return False

        forward_pref = self._entry_preference(forward_entry)
        reverse_pref = self._entry_preference(reverse_entry)
        if forward_pref is None or reverse_pref is None:
            return False

        # Sample which direction to respect.
        if random.randint(0, 1) == 1:
            return forward_pref == "B"
        return reverse_pref == "A"
