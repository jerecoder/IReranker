from __future__ import annotations

from ireranker.oracles import (
    BidirectionalMatrixOracle,
    CachedSamplingMatrixOracle,
    Oracle,
    SamplingMatrixOracle,
    WeirdSamplingMatrixOracle,
)

from .pac_optimized import PACOptimizedRanker
from .registry import register_ranker


@register_ranker(
    "PAC + Bubble",
    oracle_factories=[
        ("sampling", lambda seed: SamplingMatrixOracle(seed=seed)),
        ("cached-sampling", lambda seed: CachedSamplingMatrixOracle(seed=seed)),
        ("bidirectional", lambda seed: BidirectionalMatrixOracle()),
        (
            "weird $(1.5)$",
            lambda seed: WeirdSamplingMatrixOracle(seed=seed, expected_samples=1.5),
        ),
    ],
)
class PACBubbleRanker(PACOptimizedRanker):
    """PAC Top-k with Bubble Sort refinement.

    Strategy:
    1. Use PAC (Optimized) to select top-k set (fast, ~150 comparisons)
    2. Order selected set by BM25 prior
    3. Refine ordering with bubble sort (adds ~45 comparisons for K=10)

    Total: ~195 comparisons for high-quality top-10 ranking
    """

    def __init__(
        self,
        oracle: Oracle,
        seed: int | None = None,
        top_k: int = 10,
        candidate_pool_multiplier: int = 3,
        num_child: int = 1,
    ):
        super().__init__(
            oracle=oracle,
            seed=seed,
            top_k=top_k,
            candidate_pool_multiplier=candidate_pool_multiplier,
            bubble_refine=True,  # Always enable bubble refinement
            num_child=num_child,
        )
