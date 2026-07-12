from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, TypeVar

from ireranker.oracles import BudgetExceeded


Item = TypeVar("Item")
ChooseBest = Callable[[Sequence[Item]], Item]


def strided_groups(n: int, top_k: int) -> tuple[list[int], list[list[int]]]:
    """Match Mohajer's K strided candidate groups over an initial BM25 order."""
    if n < 0 or top_k <= 0:
        raise ValueError("n must be non-negative and top_k must be positive")
    k = min(top_k, n)
    if k == 0:
        return [], []
    indices = [
        alpha * k + rank
        for rank in range(k)
        for alpha in range(n // k + 1)
        if alpha * k + rank < n
    ]
    group_size = (n + k - 1) // k
    groups = [
        indices[group * group_size : min((group + 1) * group_size, n)]
        for group in range(k)
    ]
    if sorted(item for group in groups for item in group) != list(range(n)):
        raise RuntimeError("Strided grouping failed to partition the candidates")
    return indices, groups


def tournament_winner(
    items: Sequence[Item],
    *,
    arity: int,
    choose_best: ChooseBest[Item],
) -> Item:
    if arity < 2:
        raise ValueError("arity must be at least two")
    current = list(items)
    if not current:
        raise ValueError("cannot select a winner from an empty sequence")
    while len(current) > 1:
        next_round: list[Item] = []
        for start in range(0, len(current), arity):
            match = current[start : start + arity]
            if len(match) == 1:
                next_round.append(match[0])
                continue
            winner = choose_best(match)
            if winner not in match:
                raise ValueError("chooser returned an item outside its match")
            next_round.append(winner)
        current = next_round
    return current[0]


@dataclass
class _Champion:
    item: int
    group: int


def active_multiway_topk(
    n: int,
    *,
    top_k: int,
    arity: int,
    choose_best: ChooseBest[int],
) -> list[int]:
    """Mohajer-style group tournaments plus a multiway champion heap."""
    indices, groups = strided_groups(n, top_k)
    if not indices:
        return []
    ranking: list[int] = []
    heap: list[_Champion] = []

    def heapify(position: int) -> None:
        child_start = arity * position + 1
        child_positions = list(range(child_start, min(child_start + arity, len(heap))))
        positions = [position] + child_positions
        if len(positions) <= 1:
            return
        winner_item = choose_best([heap[index].item for index in positions])
        winner_position = next(
            index for index in positions if heap[index].item == winner_item
        )
        if winner_position != position:
            heap[position], heap[winner_position] = heap[winner_position], heap[position]
            heapify(winner_position)

    try:
        for group_id, group in enumerate(groups):
            if not group:
                continue
            winner = tournament_winner(group, arity=arity, choose_best=choose_best)
            heap.append(_Champion(winner, group_id))
        for position in range((len(heap) - 2) // arity, -1, -1):
            heapify(position)

        k = min(top_k, n)
        while heap and len(ranking) < k:
            champion = heap[0]
            ranking.append(champion.item)
            groups[champion.group] = [
                item for item in groups[champion.group] if item != champion.item
            ]
            if groups[champion.group]:
                replacement = tournament_winner(
                    groups[champion.group], arity=arity, choose_best=choose_best
                )
                heap[0] = _Champion(replacement, champion.group)
            else:
                heap[0] = heap[-1]
                heap.pop()
            if heap:
                heapify(0)
    except BudgetExceeded:
        pass

    seen = set(ranking)
    heap_tail = [champion.item for champion in heap if champion.item not in seen]
    seen.update(heap_tail)
    leftovers = [item for item in indices if item not in seen]
    result = ranking + heap_tail + leftovers
    if len(result) != n or set(result) != set(range(n)):
        raise RuntimeError("Active multiway scheduler failed to return a full permutation")
    return result


def standard_multiway_heapsort_topk(
    items: Sequence[Item],
    *,
    top_k: int,
    arity: int,
    choose_best: ChooseBest[Item],
) -> list[Item]:
    """Standard d-ary partial heapsort using the identical multiway chooser."""
    original = list(items)
    order = list(items)
    extracted: list[Item] = []

    def heapify(size: int, position: int) -> None:
        child_start = arity * position + 1
        child_positions = list(range(child_start, min(child_start + arity, size)))
        positions = [position] + child_positions
        if len(positions) <= 1:
            return
        winner = choose_best([order[index] for index in positions])
        winner_position = next(index for index in positions if order[index] == winner)
        if winner_position != position:
            order[position], order[winner_position] = order[winner_position], order[position]
            heapify(size, winner_position)

    try:
        size = len(order)
        for position in range((size - 2) // arity, -1, -1):
            heapify(size, position)
        for end in range(size - 1, 0, -1):
            order[end], order[0] = order[0], order[end]
            extracted.append(order[end])
            if len(extracted) >= min(top_k, size):
                break
            heapify(end, 0)
    except BudgetExceeded:
        pass
    seen = set(extracted)
    result = extracted + [item for item in original if item not in seen]
    if len(result) != len(original) or set(result) != set(original):
        raise RuntimeError("Standard multiway heapsort failed to return a full permutation")
    return result
