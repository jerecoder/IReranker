from __future__ import annotations

from typing import List

from .ranker import CacheRanker


# @register_ranker("PRP-Allpair (Win-Ratio)")
class PRPAllpairRanker(CacheRanker):
    """
    Paper-aligned PRP-Allpair (Qin et al. 2024).

    - Uses PRP pairwise comparisons as in §3.1–3.2.
    - For each unordered pair (i, j), calls the oracle twice:
        * lt(i, j): True if i should be AFTER j.
        * lt(j, i): True if j should be AFTER i.
      and aggregates wins / ties into per-document scores.
    - Score(i) = (#wins + 0.5 * #ties) over all j != i.
      Normalization by (n-1) is omitted since it does not affect ranking.
    - Final ranking: sort by descending score, with ties broken by
      the initial index (BM25 order), i.e., BM25 is used to resolve
      prediction conflicts or failures, as in the paper.
    """

    def __init__(self, oracle, seed: int | None = None) -> None:
        super().__init__(oracle, seed)

    def _rank(self) -> List[int]:
        n = self.n
        if n <= 1:
            return list(range(n))

        # Initial order corresponds to BM25 order because the BEIR
        # pipeline reorders candidate_ids before calling rank().
        initial_order = list(range(n))

        # Win/tie scores
        scores = [0.0] * n

        # Enumerate all unordered pairs (i, j), i < j
        for i in range(n):
            for j in range(i + 1, n):
                # PRP comparisons:
                # lt(a, b) == True  ⇒  a should be ranked AFTER b (b is better).
                a_after_b = self.lt(i, j)
                b_after_a = self.lt(j, i)

                if a_after_b and not b_after_a:
                    # j is better than i
                    scores[j] += 1.0
                elif b_after_a and not a_after_b:
                    # i is better than j
                    scores[i] += 1.0
                else:
                    # Inconsistent / tie / missing entries:
                    # each doc gets half a point, per §3.2.
                    scores[i] += 0.5
                    scores[j] += 0.5

        # Sort by descending score; break ties by initial BM25 order
        order = sorted(
            initial_order,
            key=lambda idx: (-scores[idx], idx),
        )
        return order
