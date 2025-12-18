from __future__ import annotations

import math
import random
from typing import List

from ireranker.oracles import (
    BidirectionalMatrixOracle,
    BudgetExceeded,
    CachedSamplingMatrixOracle,
    Oracle,
    SamplingMatrixOracle,
    WeirdSamplingMatrixOracle,
)

from .ranker import Ranker
from .registry import register_ranker


# @register_ranker(
#     "PAC Top-k (3-round)",
#     oracle_factories=[
#         ("sampling", lambda seed: SamplingMatrixOracle(seed=seed)),
#         ("cached-sampling", lambda seed: CachedSamplingMatrixOracle(seed=seed)),
#         ("bidirectional", lambda seed: BidirectionalMatrixOracle()),
#         (
#             "weird $(1.5)$",
#             lambda seed: WeirdSamplingMatrixOracle(seed=seed, expected_samples=1.5),
#         ),
#     ],
# )
class PACRanker(Ranker):
    """PAC Top-k Identification following Agarwal et al.

    Implements the 3-round algorithm with:
    - Round 1: Select best k anchors from random sample
    - Round 2: Compare anchors vs all items, build winner sets
    - Round 3: Refine critical set

    Uses BM25 prior for:
    - Initial ordering (items are indices in BM25 order)
    - Tie-breaking when sorting anchors
    - Final ordering within the selected top-k set
    """

    def __init__(
        self,
        oracle: Oracle,
        seed: int | None = None,
        top_k: int = 10,
        epsilon: float = 0.1,
        use_prior_for_anchors: bool = True,
    ):
        """
        Args:
            oracle: Comparison oracle
            seed: Random seed
            top_k: Number of items to select
            epsilon: PAC accuracy parameter
            use_prior_for_anchors: If True, bias anchor selection towards top of BM25 order
        """
        self.k = top_k
        self.epsilon = epsilon
        self.use_prior_for_anchors = use_prior_for_anchors
        super().__init__(oracle, seed)

    def set_seed(self, seed: int | None) -> None:
        super().set_seed(seed)
        self._rng = random.Random(self.seed)

    def _rank(self) -> List[int]:
        """Execute the 3-round PAC top-k algorithm."""
        n = self.n
        K = min(self.k, n)

        if n <= K:
            return list(range(n))

        try:
            # Round 1: Select best k anchors
            anchors = self._round1_select_anchors(n, K)

            # Round 2: Build winner sets and find critical set
            top_k_set, need_refinement = self._round2_winner_sets(anchors, K)

            # Round 3: Refine if needed
            if need_refinement:
                top_k_set = self._round3_refine(top_k_set, K)

            # Sort the top-k set by BM25 prior (orig0nal index order)
            top_k_set = sorted(top_k_set)

            # Append remaining items in BM25 order
            remaining = [i for i in range(n) if i not in top_k_set]
            return top_k_set + remaining

        except BudgetExceeded:
            # Best effort: return BM25 order
            return list(range(n))

    def _round1_select_anchors(self, n: int, K: int) -> List[int]:
        """Round 1: Sample anchors and select best K.

        Sample O(sqrt(n)) anchors and compare them pairwise to find
        the best K anchors (those not strongly beaten by others).
        Uses BM25 prior for tie-breaking.
        """
        # Sample anchor candidates
        if self.use_prior_for_anchors:
            # Bias sampling towards top of BM25 order
            # Sample from top 2*sqrt(n)*K items preferentially
            pool_size = min(n, max(K * 4, int(2 * math.sqrt(n) * K)))
            anchor_pool = list(range(pool_size))
            num_anchors = min(pool_size, max(K * 2, int(math.sqrt(n))))
            anchors_raw = self._rng.sample(anchor_pool, num_anchors)
        else:
            # Uniform random sampling
            num_anchors = min(n, max(K * 2, int(math.sqrt(n))))
            anchors_raw = self._rng.sample(range(n), num_anchors)

        # Compare anchors pairwise to estimate quality
        # Build estimated pairwise preference matrix for anchors
        anchor_beats = {a: 0 for a in anchors_raw}

        for i, a1 in enumerate(anchors_raw):
            for a2 in anchors_raw[i + 1 :]:
                if self.gt(a1, a2):  # a1 > a2 (a1 is better)
                    anchor_beats[a1] += 1
                else:
                    anchor_beats[a2] += 1

        # Select top K anchors by win count, use BM25 order (index) for ties
        sorted_anchors = sorted(
            anchors_raw,
            key=lambda a: (-anchor_beats[a], a),  # wins desc, index asc
        )
        return sorted_anchors[:K]

    def _round2_winner_sets(self, anchors: List[int], K: int) -> tuple[List[int], bool]:
        """Round 2: Compare anchors vs all items, build winner sets.

        Returns:
            (top_k_set, need_refinement):
                - If we can confidently identify top-k, return (set, False)
                - Otherwise return (partial_set, True) and need round 3
        """
        n = self.n

        # Build winner sets: for each anchor, find items that beat it
        winner_sets = {}
        for anchor in anchors:
            winners = []
            for item in range(n):
                if item == anchor:
                    continue
                # Check if item beats anchor with margin
                if self.gt(item, anchor):  # item > anchor
                    winners.append(item)
            winner_sets[anchor] = set(winners)

        # Sort anchors by quality
        # Use: (1) pairwise comparison if clear, (2) winner set size, (3) BM25 order
        def anchor_key(a: int) -> tuple:
            return (len(winner_sets[a]), a)  # smaller winner set = better, then BM25

        sorted_anchors = sorted(anchors, key=anchor_key)

        # Greedy construction: add winner sets until we have k items
        top_k_candidates = set()

        for anchor in sorted_anchors:
            # Also check items that beat earlier (better) anchors
            for earlier_anchor in sorted_anchors:
                if earlier_anchor == anchor:
                    break
                # Items beating earlier anchors are definitely good
                top_k_candidates.update(winner_sets[earlier_anchor])

            if len(top_k_candidates) >= K:
                break

        if len(top_k_candidates) >= K:
            # We have enough clear winners, take top K by BM25 order
            return sorted(list(top_k_candidates))[:K], False
        else:
            # Need refinement - not enough clear winners
            return list(top_k_candidates), True

    def _round3_refine(self, partial_set: List[int], K: int) -> List[int]:
        """Round 3: Refine the selection by examining remaining candidates.

        We have a partial set of size < K. Need to find K - len(partial_set) more items
        from the remaining candidates.
        """
        n = self.n
        partial_size = len(partial_set)
        needed = K - partial_size

        if needed <= 0:
            return partial_set

        # Candidates are items not yet in partial_set
        candidates = [i for i in range(n) if i not in partial_set]

        if len(candidates) <= needed:
            # Just take all candidates
            return partial_set + candidates

        # Compare candidates against each other to find best ones
        # Use a simple tournament: compare each candidate against a sample of others
        candidate_scores = {c: 0 for c in candidates}

        # Sample-based comparison: for each candidate, compare against sqrt(len(candidates)) others
        sample_size = min(len(candidates), max(10, int(math.sqrt(len(candidates)))))

        for candidate in candidates:
            opponents = self._rng.sample(
                [c for c in candidates if c != candidate],
                min(sample_size, len(candidates) - 1),
            )
            for opponent in opponents:
                if self.gt(candidate, opponent):
                    candidate_scores[candidate] += 1

        # Select top 'needed' candidates by score, use BM25 order for ties
        sorted_candidates = sorted(candidates, key=lambda c: (-candidate_scores[c], c))

        return partial_set + sorted_candidates[:needed]

    def _better(self, i: int, j: int) -> bool:
        """Check if item i is better than item j.

        For now, simplified to just use gt comparison.
        In a full implementation with repeated sampling, this would
        estimate P(i > j) and check if it exceeds 0.5 + epsilon.
        """
        return self.gt(i, j)


# @register_ranker(
#     "PAC Top-k (2-round)",
#     oracle_factories=[
#         ("sampling", lambda seed: SamplingMatrixOracle(seed=seed)),
#         ("cached-sampling", lambda seed: CachedSamplingMatrixOracle(seed=seed)),
#         ("bidirectional", lambda seed: BidirectionalMatrixOracle()),
#         (
#             "weird $(1.5)$",
#             lambda seed: WeirdSamplingMatrixOracle(seed=seed, expected_samples=1.5),
#         ),
#     ],
# )
class PAC2RoundRanker(Ranker):
    """Simplified 2-round PAC Top-k algorithm.

    Round 1: Sample anchors and compare all items to all anchors
    Round 2: Refine critical set if needed

    Simpler than 3-round but uses O(n^(4/3) + nk) comparisons.
    """

    def __init__(
        self,
        oracle: Oracle,
        seed: int | None = None,
        top_k: int = 10,
        epsilon: float = 0.1,
        use_prior_for_anchors: bool = True,
    ):
        self.k = top_k
        self.epsilon = epsilon
        self.use_prior_for_anchors = use_prior_for_anchors
        super().__init__(oracle, seed)

    def set_seed(self, seed: int | None) -> None:
        super().set_seed(seed)
        self._rng = random.Random(self.seed)

    def _rank(self) -> List[int]:
        """Execute the 2-round PAC top-k algorithm."""
        n = self.n
        K = min(self.k, n)

        if n <= K:
            return list(range(n))

        try:
            # Round 1: Sample anchors and build winner sets
            num_anchors = min(n, max(K, int(n ** (1 / 3))))

            if self.use_prior_for_anchors:
                # Sample from top of BM25 order with bias
                pool_size = min(n, num_anchors * 3)
                anchors = self._rng.sample(range(pool_size), num_anchors)
            else:
                anchors = self._rng.sample(range(n), num_anchors)

            # Compare all items to all anchors
            # For each anchor, find its winner set
            winner_sets = {}
            for anchor in anchors:
                winners = []
                for item in range(n):
                    if item != anchor and self.gt(item, anchor):
                        winners.append(item)
                winner_sets[anchor] = set(winners)

            # Sort anchors by winner set size (smaller = better)
            sorted_anchors = sorted(anchors, key=lambda a: (len(winner_sets[a]), a))

            # Greedy construction
            top_k_set = set()
            critical_anchor = None

            for anchor in sorted_anchors:
                candidates = winner_sets[anchor]
                if len(top_k_set) + len(candidates) <= K:
                    top_k_set.update(candidates)
                else:
                    critical_anchor = anchor
                    break

            # Round 2: Refine if we don't have exactly K items yet
            if len(top_k_set) < K and critical_anchor is not None:
                needed = K - len(top_k_set)
                candidates = [c for c in winner_sets[critical_anchor] if c not in top_k_set]

                if len(candidates) <= needed:
                    top_k_set.update(candidates)
                else:
                    # Compare candidates pairwise
                    candidate_scores = {c: 0 for c in candidates}
                    for i, c1 in enumerate(candidates):
                        for c2 in candidates[i + 1 :]:
                            if self.gt(c1, c2):
                                candidate_scores[c1] += 1
                            else:
                                candidate_scores[c2] += 1

                    sorted_candidates = sorted(candidates, key=lambda c: (-candidate_scores[c], c))
                    top_k_set.update(sorted_candidates[:needed])

            # Convert to sorted list (by BM25 order)
            result = sorted(list(top_k_set))
            remaining = [i for i in range(n) if i not in result]
            return result + remaining

        except BudgetExceeded:
            return list(range(n))
