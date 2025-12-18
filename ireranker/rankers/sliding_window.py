from __future__ import annotations

from typing import List

from ireranker.oracles import (
    Oracle,
)

from .ranker import Ranker


# @register_ranker(
#     "Sliding Window PRP (Classic)",
#     oracle_factories=[
#         ("bidirectional", lambda seed: BidirectionalMatrixOracle()),
#         ("cached-sampling", lambda seed: CachedSamplingMatrixOracle(seed=seed)),
#         ("sampling", lambda seed: SamplingMatrixOracle(seed=seed)),
#         (
#             "weird $(1.5)$",
#             lambda seed: WeirdSamplingMatrixOracle(seed=seed, expected_samples=1.5),
#         ),
#     ],
# )
class SlidingWindowRanker(Ranker):
    """PRP-Sliding-K: sliding-window passes starting from the bottom of the list.

    One pass is a single bottom-up sweep over adjacent pairs. We perform a
    small, fixed number of passes (K), which is typically enough to get a
    good Top-K ranking while keeping complexity close to O(K · N) with
    K ≪ N.
    """

    def __init__(
        self,
        oracle: Oracle,
        seed: int | None = None,
        passes: int = 10,
    ):
        super().__init__(oracle, seed)
        self.passes = passes

    def _rank(self) -> List[int]:
        n = self.n
        order = list(range(n))

        if n <= 1:
            return order

        num_passes = min(self.passes, n - 1)
        for _ in range(num_passes):
            swapped = False

            for i in range(n - 2, -1, -1):
                if self.lt(order[i], order[i + 1]):
                    order[i], order[i + 1] = order[i + 1], order[i]
                    swapped = True

            if not swapped:
                break

        return order
