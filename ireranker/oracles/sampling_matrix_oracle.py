from __future__ import annotations

import random

from .oracle import MatrixOracle


class SamplingMatrixOracle(MatrixOracle):
    """Oracle that samples direction when both forward/reverse preferences exist."""

    def __init__(self, base_dir=None, seed: int | None = None):
        super().__init__(base_dir=base_dir, cache_comparisons=False)
        self._rng = random.Random(seed)

    def set_seed(self, seed: int | None) -> None:
        super().set_seed(seed)
        self._rng = random.Random(seed)

    def sample_lt(self, i: int, j: int) -> bool:
        matrix = self._ensure_matrix_loaded()

        if self.current_task is None:
            return False

        qid = self.current_task.query_id
        doc_a = self.current_task.candidate_ids[i]
        doc_b = self.current_task.candidate_ids[j]

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
        if self._rng.randint(0, 1) == 1:
            return forward_pref == "B"
        return reverse_pref == "A"
