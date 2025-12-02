from __future__ import annotations

import math
import random

from .oracle import MatrixOracle


class SamplingMatrixOracle(MatrixOracle):
    """Oracle that samples direction when both forward/reverse preferences exist."""

    def __init__(
        self,
        base_dir=None,
        seed: int | None = None,
        *,
        cache_comparisons: bool = False,
    ):
        super().__init__(base_dir=base_dir, cache_comparisons=cache_comparisons)
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


class CachedSamplingMatrixOracle(SamplingMatrixOracle):
    """Sampling oracle that reuses sampled outcomes via comparison caching."""

    def __init__(self, base_dir=None, seed: int | None = None):
        super().__init__(base_dir=base_dir, seed=seed, cache_comparisons=True)


class WeirdSamplingMatrixOracle(MatrixOracle):
    """Oracle that samples direction when both forward/reverse preferences exist."""

    def __init__(
        self,
        base_dir=None,
        seed: int | None = None,
        expected_samples: float = 1.0,
    ):
        super().__init__(base_dir=base_dir, cache_comparisons=False)
        self._rng = random.Random(seed)
        self.expected_samples = max(expected_samples, 0.0)

    def set_seed(self, seed: int | None) -> None:
        super().set_seed(seed)
        self._rng = random.Random(seed)

    def sample_lt(self, i: int, j: int) -> bool:
        forward_pref, reverse_pref = self._pair_preferences(i, j)
        if forward_pref is None or reverse_pref is None:
            return False
        if self.expected_samples < 1.0 and self._rng.random() > self.expected_samples:
            return False

        o1, o2 = 1 - 2 * int(forward_pref == "B"), 1 - 2 * int(reverse_pref == "A")
        if o1 == o2:
            return o1 == -1
        if self._rng.random() > 0.5:
            o1, o2 = o2, o1
        one_more = int(
            self._rng.random() < (self.expected_samples - 1.0)
        )  # super important: it assumes that 1 <= m <= 2
        return (o1 + one_more * (o2)) > 0
