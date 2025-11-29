from __future__ import annotations

import math
import random

from .oracle import MatrixOracle


class SamplingMatrixOracle(MatrixOracle):
    """Oracle that samples direction when both forward/reverse preferences exist."""

    def __init__(self, base_dir=None, seed: int | None = None):
        super().__init__(base_dir=base_dir, cache_comparisons=False)
        self._rng = random.Random(seed)
        self.E_s = 1.0

    def set_seed(self, seed: int | None) -> None:
        super().set_seed(seed)
        self._rng = random.Random(seed)

    def sample_lt(self, i: int, j: int) -> bool:
        forward_pref, reverse_pref = self._pair_preferences(i, j)
        if forward_pref is None or reverse_pref is None:
            return False

        o1, o2 = 1 - 2 * int(forward_pref == "B"), 1 - 2 * int(reverse_pref == "A")
        if o1 == o2:
            return o1 == -1
        one_more = int(self._rng.random() < (self.E_s - math.floor(self.E_s)))
        return (self._rng.choice([o1, o2]) + one_more * (self._rng.choice([o1, o2]))) > 0


class WeirdSamplingMatrixOracle(MatrixOracle):
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
