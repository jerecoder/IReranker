from __future__ import annotations

from typing import List

from ireranker.oracles import (
    BidirectionalMatrixOracle,
    BudgetExceeded,
    Oracle,
    SamplingMatrixOracle,
)

from .bubble_ranker import BubbleRanker
from .mohajer_ranker import MohajerRanker
from .registry import register_ranker


@register_ranker(
    "Mohajer + Bubble",
    oracle_factories=[
        ("bidirectional", lambda seed: BidirectionalMatrixOracle()),
        ("sampling", lambda seed: SamplingMatrixOracle(seed=seed)),
    ],
)
class MohajerBubbleRanker(BubbleRanker):
    """Seed with Mohajer ranking, then refine locally with bubble sort."""

    def __init__(
        self,
        oracle: Oracle,
        seed: int | None = None,
        *,
        num_child: int = 1,
        top_k: int | None = None,
    ):
        # Default to top_k=10 for Mohajer+Bubble (since we only care about top-10 this may need to change for other evaluations)
        if top_k is None:
            top_k = 10
        self.k_mult = 2
        super().__init__(oracle, seed, num_child=num_child, top_k=top_k)
        # Keep Mohajer focused on top_k comparisons; we will later reuse its heap
        # ordering to extend the prefix without further oracle calls.
        self._mohajer = MohajerRanker(oracle=oracle, seed=seed, top_k=top_k)

    def set_seed(self, seed: int | None) -> None:
        super().set_seed(seed)
        if hasattr(self, "_mohajer"):
            self._mohajer.set_seed(seed)

    def _rank(self) -> List[int]:
        # Start with Mohajer's strong initial ordering.
        mohajer_order = self._mohajer.rank(self.task)
        n = len(mohajer_order)
        if n <= 1:
            return mohajer_order

        # Only refine the prefix where we expect the true top-k to live (top_k).
        k = self._effective_k(n)
        prefix_size = min(n, k * self.k_mult)
        prefix = mohajer_order[:prefix_size]
        suffix = mohajer_order[prefix_size:]

        try:
            self._bubble_refine(prefix)
        except BudgetExceeded:
            pass

        return prefix + suffix

    def _bubble_refine(self, ranking: List[int]) -> None:
        """Run Bubble Sort in-place on the provided ranking."""
        n = len(ranking)
        if n <= 1:
            return

        window = self.num_child + 1
        # Use the original top_k (based on full task size), not the prefix size
        k = self._effective_k(self.n)
        last_start = len(ranking) - window

        for i in range(k):
            start_ind = last_start
            end_ind = last_start + window
            is_change = False
            while True:
                if start_ind < i:
                    start_ind = i
                if end_ind > len(ranking):
                    end_ind = len(ranking)
                if end_ind <= start_ind:
                    end_ind = start_ind + 1

                best_offset = self._best_in_window(ranking, start_ind, end_ind)
                if best_offset != 0:
                    best_idx = start_ind + best_offset
                    ranking[start_ind], ranking[best_idx] = (
                        ranking[best_idx],
                        ranking[start_ind],
                    )
                    if not is_change:
                        is_change = True
                        if (
                            last_start != len(ranking) - window
                            and best_offset == len(ranking[start_ind:end_ind]) - 1
                        ):
                            last_start += len(ranking[start_ind:end_ind]) - 1

                if start_ind == i:
                    break

                if not is_change:
                    last_start -= self.num_child

                start_ind -= self.num_child
                end_ind -= self.num_child
