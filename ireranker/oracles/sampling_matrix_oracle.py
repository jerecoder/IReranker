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
        forward_pref, reverse_pref = self._pair_preferences(i, j)
        if forward_pref is None or reverse_pref is None:
            return False

        # Sample which direction to respect.
        if self._rng.randint(0, 1) == 1:
            return forward_pref == "B"
        return reverse_pref == "A"
