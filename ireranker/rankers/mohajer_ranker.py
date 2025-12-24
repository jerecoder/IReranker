from __future__ import annotations

import heapq
import math
import os
import random
from typing import List

from ireranker.oracles import (
    BudgetExceeded,
    BidirectionalFlanSeq2SeqOracle,
    Oracle,
    SamplingMatrixOracle,
)

from .ranker import Ranker
from .registry import register_ranker


@register_ranker(
    "Mohajer (IR)",
    oracle_factories=[
        ("sampling", lambda seed: SamplingMatrixOracle(seed=seed)),
        (
            "flan-live",
            lambda seed: BidirectionalFlanSeq2SeqOracle(
                model_name=os.environ.get("FLAN_MODEL_NAME", "google/flan-t5-xl"),
                quantization=os.environ.get("FLAN_QUANT", "8bit"),
                device=os.environ.get("FLAN_DEVICE", "cuda"),
            ),
        ),
    ],
)
class MohajerRanker(Ranker):
    def __init__(self, oracle: Oracle, seed: int | None = None, top_k: int = 10):
        self.k = top_k
        self.m = 1.0
        super().__init__(oracle, seed)

    def set_seed(self, seed: int | None) -> None:
        super().set_seed(seed)
        self._rng = random.Random(self.seed)

    def select_winner(self, indices: list[int]):
        order = list(indices)

        while len(order) > 1:
            new_order: list[int] = []

            for k in range(0, len(order) - 1, 2):
                i = order[k]
                j = order[k + 1]

                if self._better(i, j):
                    new_order.append(i)
                else:
                    new_order.append(j)

            if len(order) % 2 == 1:
                new_order.append(order[-1])

            order = new_order

        return order[0]

    def _get_indices(self):
        indices = []
        for r in range(self.k):
            for idx in [alpha * self.k + r for alpha in range(self.n // self.k + 1)]:
                if idx < self.n:
                    indices.append(idx)
        return indices

    def _rank(self) -> List[int]:
        # number of groups / desired top-K
        K = min(self.k, self.n)

        # group size Q = ceil(n / K)
        Q = (self.n + K - 1) // K

        # Mohajer benefits from randomized initial ordering; shuffle once per run.
        indices = self._get_indices()
        groups: list[list[int]] = []
        champions: list[int | None] = []

        try:
            # 1) build groups and find each group's champion using SELECT
            for g in range(K):
                start = g * Q
                end = min((g + 1) * Q, self.n)
                group_indices = indices[start:end]
                groups.append(group_indices)

                if group_indices:
                    champ = self.select_winner(group_indices)
                    champions.append(champ)
                else:
                    champions.append(None)

            # 2) build winners heap (heap of champions)
            winner_heap: list[_HeapItem] = []
            champ_to_group: dict[int, int] = {}

            for g, champ in enumerate(champions):
                if champ is not None:
                    heap_item = _HeapItem(self, champ)
                    heapq.heappush(winner_heap, heap_item)
                    champ_to_group[champ] = g

            if not winner_heap:
                return []

            # 3) repeatedly pop best champion, then refill from its home group using SELECT
            ranking: list[int] = []

            while winner_heap and len(ranking) < K:
                best_item = heapq.heappop(winner_heap)
                best_item_idx = best_item.idx
                ranking.append(best_item_idx)

                champ_og_group = champ_to_group.pop(best_item_idx)

                # remove this index from its group
                groups[champ_og_group] = [
                    idx for idx in groups[champ_og_group] if idx != best_item_idx
                ]

                # if group still has items, compute new champion and push into heap
                if groups[champ_og_group]:
                    new_champ = self.select_winner(groups[champ_og_group])
                    champ_to_group[new_champ] = champ_og_group
                    heapq.heappush(winner_heap, _HeapItem(self, new_champ))

            # Heap still encodes priorities among the remaining group champions; emit them
            # (in heap array order) ahead of the untouched leftovers without extra oracle calls.
            heap_tail = [item.idx for item in winner_heap]

            ranked_set = set(ranking)
            ranked_set.update(heap_tail)
            leftovers = [idx for idx in indices if idx not in ranked_set]
            return ranking + heap_tail + leftovers

        except BudgetExceeded:
            # Construct best-effort ranking
            heap_tail = [item.idx for item in winner_heap] if "winner_heap" in locals() else []
            current_ranking = ranking if "ranking" in locals() else []

            # Collect all other items
            ranked_set = set(current_ranking)
            ranked_set.update(heap_tail)

            # Flatten remaining groups if they exist
            group_leftovers = []
            if "groups" in locals():
                for g_list in groups:
                    for idx in g_list:
                        if idx not in ranked_set:
                            group_leftovers.append(idx)

            # Also check indices that might not have been put into groups yet (if error happened in step 1)
            # But indices is local.
            all_leftovers = [
                idx for idx in indices if idx not in ranked_set and idx not in group_leftovers
            ]

            return current_ranking + heap_tail + group_leftovers + all_leftovers

    def _better(self, i: int, j: int) -> bool:
        full = math.floor(self.m)
        frac = self.m - full
        wins = sum(not self.lt(i, j) for _ in range(full))
        total = full
        if frac > 0 and self._rng.random() < frac:
            wins += not self.lt(i, j)
            total += 1
        return wins >= total / 2


@register_ranker(
    "Jingle Bells",
    oracle_factories=[
        ("sampling", lambda seed: SamplingMatrixOracle(seed=seed)),
        (
            "flan-live",
            lambda seed: BidirectionalFlanSeq2SeqOracle(
                model_name=os.environ.get("FLAN_MODEL_NAME", "google/flan-t5-xl"),
                quantization=os.environ.get("FLAN_QUANT", "8bit"),
                device=os.environ.get("FLAN_DEVICE", "cuda"),
            ),
        ),
    ],
)
class MohajerBM25Ranker(MohajerRanker):
    """Mohajer variant that consumes BM25-ordered slices instead of in-group tournaments."""

    def __init__(self, oracle: Oracle, seed: int | None = None, top_k: int = 10):
        super().__init__(oracle, seed, top_k=top_k)

    def _rank(self) -> List[int]:
        if self.n == 0:
            return []

        K = min(self.k, self.n)
        Q = (self.n + K - 1) // K

        indices = self._get_indices()
        groups: list[list[int]] = []
        next_in_group: list[int] = []
        ranking: list[int] = []

        try:
            # Partition candidates into strided BM25-ordered groups.
            for g in range(K):
                start = g * Q
                end = min((g + 1) * Q, self.n)
                group_indices = indices[start:end]
                groups.append(group_indices)
                next_in_group.append(0)

            # Initialize heap with the BM25 head from each group.
            winner_heap: list[_HeapItem] = []
            champ_to_group: dict[int, int] = {}

            for g, group_indices in enumerate(groups):
                if not group_indices:
                    continue

                champ = group_indices[0]
                next_in_group[g] = 1
                heapq.heappush(winner_heap, _HeapItem(self, champ))
                champ_to_group[champ] = g

            if not winner_heap:
                return []

            # Merge groups using the same Mohajer heap logic.
            while winner_heap and len(ranking) < K:
                best_item = heapq.heappop(winner_heap)
                best_item_idx = best_item.idx
                ranking.append(best_item_idx)

                champ_group = champ_to_group.pop(best_item_idx)

                if next_in_group[champ_group] < len(groups[champ_group]):
                    new_champ = groups[champ_group][next_in_group[champ_group]]
                    next_in_group[champ_group] += 1
                    champ_to_group[new_champ] = champ_group
                    heapq.heappush(winner_heap, _HeapItem(self, new_champ))

            heap_tail = [item.idx for item in winner_heap]

            ranked_set = set(ranking)
            ranked_set.update(heap_tail)

            leftovers: list[int] = []
            for g, group_indices in enumerate(groups):
                start_at = next_in_group[g]
                for idx in group_indices[start_at:]:
                    if idx not in ranked_set:
                        leftovers.append(idx)

            for idx in indices:
                if idx not in ranked_set and idx not in leftovers:
                    leftovers.append(idx)

            return ranking + heap_tail + leftovers

        except BudgetExceeded:
            heap_tail = [item.idx for item in winner_heap] if "winner_heap" in locals() else []
            ranked_set = set(ranking)
            ranked_set.update(heap_tail)

            leftovers: list[int] = []
            if "groups" in locals():
                for g, group_indices in enumerate(groups):
                    start_at = next_in_group[g] if g < len(next_in_group) else 0
                    for idx in group_indices[start_at:]:
                        if idx not in ranked_set:
                            leftovers.append(idx)

            if "indices" in locals():
                for idx in indices:
                    if idx not in ranked_set and idx not in leftovers:
                        leftovers.append(idx)

            return ranking + heap_tail + leftovers


@register_ranker(
    "Christmas Tree",
    oracle_factories=[
        ("sampling", lambda seed: SamplingMatrixOracle(seed=seed)),
        (
            "flan-live",
            lambda seed: BidirectionalFlanSeq2SeqOracle(
                model_name=os.environ.get("FLAN_MODEL_NAME", "google/flan-t5-xl"),
                quantization=os.environ.get("FLAN_QUANT", "8bit"),
                device=os.environ.get("FLAN_DEVICE", "cuda"),
            ),
        ),
    ],
)
class ChristmasTreeRanker(MohajerRanker):
    """Ensemble of J Mohajer-BM25 runs with random BM25-sorted groups, averaged positions."""

    def __init__(
        self,
        oracle: Oracle,
        seed: int | None = None,
        *,
        num_bells: int = 3,
        num_groups: int | None = None,
        top_k: int = 10,
    ):
        self.num_bells = max(1, num_bells)
        self.num_groups = num_groups
        super().__init__(oracle, seed, top_k=top_k)

    def set_seed(self, seed: int | None) -> None:
        super().set_seed(seed)
        self._bell_seed_rng = random.Random(self.seed)

    def _rank(self) -> List[int]:
        if self.n == 0:
            return []

        k_eff = min(self.k, self.n)
        group_count = self._effective_group_count(k_eff)
        bm25_order = list(range(self.n))
        bm25_pos = {idx: pos for pos, idx in enumerate(bm25_order)}

        per_rankings: list[list[int]] = []
        bell_seeds = [self._bell_seed_rng.randint(0, 2**31 - 1) for _ in range(self.num_bells)]

        for bell_seed in bell_seeds:
            bell_rng = random.Random(bell_seed)
            groups = self._build_random_groups(bm25_order, group_count, bell_rng)
            ranking = self._merge_groups(groups, k_eff)
            per_rankings.append(ranking)

        scores = [0.0 for _ in range(self.n)]
        counts = [0 for _ in range(self.n)]

        for ranking in per_rankings:
            for pos, idx in enumerate(ranking):
                scores[idx] += pos
                counts[idx] += 1

        avg_pos = [
            scores[idx] / counts[idx] if counts[idx] else float("inf") for idx in range(self.n)
        ]

        final_order = sorted(range(self.n), key=lambda idx: (avg_pos[idx], bm25_pos[idx]))
        return final_order

    def _effective_group_count(self, k_eff: int) -> int:
        if self.num_groups is None or self.num_groups <= 0:
            return max(1, k_eff)
        return max(1, min(self.num_groups, self.n))

    def _build_random_groups(
        self, bm25_order: list[int], group_count: int, rng: random.Random
    ) -> list[list[int]]:
        shuffled = list(bm25_order)
        rng.shuffle(shuffled)

        groups: list[list[int]] = []
        base = len(shuffled) // group_count
        remainder = len(shuffled) % group_count
        cursor = 0

        for g in range(group_count):
            size = base + (1 if g < remainder else 0)
            chunk = shuffled[cursor : cursor + size]
            cursor += size
            groups.append(sorted(chunk))

        return groups

    def _merge_groups(self, groups: list[list[int]], k_eff: int) -> list[int]:
        winner_heap: list[_HeapItem] = []
        champ_to_group: dict[int, int] = {}
        next_in_group = [1 if group else 0 for group in groups]
        ranking: list[int] = []

        try:
            for g, group in enumerate(groups):
                if not group:
                    continue
                champ = group[0]
                heapq.heappush(winner_heap, _HeapItem(self, champ))
                champ_to_group[champ] = g

            if not winner_heap:
                return []

            while winner_heap and len(ranking) < k_eff:
                best_item = heapq.heappop(winner_heap)
                best_idx = best_item.idx
                ranking.append(best_idx)

                champ_group = champ_to_group.pop(best_idx)
                if next_in_group[champ_group] < len(groups[champ_group]):
                    new_champ = groups[champ_group][next_in_group[champ_group]]
                    next_in_group[champ_group] += 1
                    champ_to_group[new_champ] = champ_group
                    heapq.heappush(winner_heap, _HeapItem(self, new_champ))

            heap_tail = [item.idx for item in winner_heap]
            ranked_set = set(ranking)
            ranked_set.update(heap_tail)

            leftovers: list[int] = []
            for g, group in enumerate(groups):
                start_at = next_in_group[g]
                for idx in group[start_at:]:
                    if idx not in ranked_set:
                        leftovers.append(idx)

            return ranking + heap_tail + leftovers

        except BudgetExceeded:
            heap_tail = [item.idx for item in winner_heap] if "winner_heap" in locals() else []
            ranked_set = set(ranking) if "ranking" in locals() else set()
            ranked_set.update(heap_tail)

            leftovers: list[int] = []
            for g, group in enumerate(groups):
                start_at = next_in_group[g] if g < len(next_in_group) else 0
                for idx in group[start_at:]:
                    if idx not in ranked_set:
                        leftovers.append(idx)

            return ranking + heap_tail + leftovers


class _HeapItem:
    """Wrap an index so that heapq uses the oracle-based comparator."""

    __slots__ = ("ranker", "task", "idx")

    def __init__(self, ranker: "MohajerRanker", idx: int):
        self.ranker = ranker
        self.idx = idx

    def __lt__(self, other: "_HeapItem") -> bool:
        return self.ranker._better(self.idx, other.idx)  # type: ignore
