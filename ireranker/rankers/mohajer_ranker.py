from __future__ import annotations

import heapq
from typing import List

from ireranker.types import Oracle, RankingTask

from .ranker import SampleRanker
from .registry import register_ranker


@register_ranker("mohajer")
class MohajerRanker(SampleRanker):
    def __init__(self, oracle: Oracle, seed: int | None = None):
        super().__init__(oracle, seed)
        self.k = 10

    def select_winner(self, task: RankingTask, indices: list[int]):
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

    def rank(self, task: RankingTask) -> List[int]:
        self.task = task
        n = len(task.candidate_ids)

        # number of groups / desired top-K
        K = min(self.k, n)

        # group size Q = ceil(n / K)
        Q = (n + K - 1) // K

        groups: list[list[int]] = []
        champions: list[int | None] = []

        # 1) build groups and find each group's champion using SELECT
        for g in range(K):
            start = g * Q
            end = min((g + 1) * Q, n)
            group_indices = list(range(start, end))
            groups.append(group_indices)

            if group_indices:
                champ = self.select_winner(task, group_indices)
                champions.append(champ)
            else:
                champions.append(None)

        # 2) build winners heap (heap of champions)
        winner_heap: list[_HeapItem] = []
        champ_to_group: dict[int, int] = {}

        for g, champ in enumerate(champions):
            if champ is not None:
                heap_item = _HeapItem(self, task, champ)
                heapq.heappush(winner_heap, heap_item)
                champ_to_group[champ] = g

        if not winner_heap:
            return []

        # 3) repeatedly pop best champion, then refill from its home group using SELECT
        ranking: list[int] = []

        for _ in range(K):
            if not winner_heap:
                break

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
                new_champ = self.select_winner(task, groups[champ_og_group])
                champ_to_group[new_champ] = champ_og_group
                heapq.heappush(winner_heap, _HeapItem(self, task, new_champ))

        return ranking + [
            idx for idx in list(range(len(task.candidate_ids))) if idx not in ranking
        ]

    def _better(self, i: int, j: int) -> bool:
        return not self.lt(
            i,
            j,
        )


class _HeapItem:
    """Wrap an index so that heapq uses the oracle-based comparator."""

    __slots__ = ("ranker", "task", "idx")

    def __init__(self, ranker: "MohajerRanker", task: RankingTask, idx: int):
        self.ranker = ranker
        self.task = task
        self.idx = idx

    def __lt__(self, other: "_HeapItem") -> bool:
        return self.ranker._better(self.idx, other.idx)  # type: ignore
