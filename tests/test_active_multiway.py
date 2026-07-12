from __future__ import annotations

import pytest

from experiments.reviewer_response.active_multiway import (
    active_multiway_topk,
    standard_multiway_heapsort_topk,
    strided_groups,
    tournament_winner,
)
from ireranker.oracles import BudgetExceeded


def test_strided_groups_match_mohajer_partition() -> None:
    indices, groups = strided_groups(100, 10)
    assert groups[0] == list(range(0, 100, 10))
    assert groups[1] == list(range(1, 100, 10))
    assert len(groups) == 10
    assert sorted(indices) == list(range(100))


def test_multiway_tournament_returns_transitive_winner() -> None:
    scores = {index: float(index) for index in range(17)}
    winner = tournament_winner(
        list(scores), arity=3, choose_best=lambda match: max(match, key=scores.get)
    )
    assert winner == 16


@pytest.mark.parametrize("arity", [2, 3, 4])
def test_active_multiway_recovers_exact_topk_under_transitive_feedback(arity: int) -> None:
    scores = {index: float(index) for index in range(100)}
    ranking = active_multiway_topk(
        100,
        top_k=10,
        arity=arity,
        choose_best=lambda match: max(match, key=scores.get),
    )
    assert ranking[:10] == list(range(99, 89, -1))
    assert len(ranking) == len(set(ranking)) == 100


def test_active_multiway_returns_permutation_when_budget_expires() -> None:
    calls = 0

    def choose(match):
        nonlocal calls
        calls += 1
        if calls > 12:
            raise BudgetExceeded("test budget")
        return max(match)

    ranking = active_multiway_topk(100, top_k=10, arity=3, choose_best=choose)
    assert len(ranking) == len(set(ranking)) == 100
    assert set(ranking) == set(range(100))


def test_standard_and_active_use_same_chooser_but_different_schedules() -> None:
    active_calls = 0
    standard_calls = 0

    def active_choose(match):
        nonlocal active_calls
        active_calls += 1
        return max(match)

    def standard_choose(match):
        nonlocal standard_calls
        standard_calls += 1
        return max(match)

    active = active_multiway_topk(100, top_k=10, arity=3, choose_best=active_choose)
    standard = standard_multiway_heapsort_topk(
        list(range(100)), top_k=10, arity=3, choose_best=standard_choose
    )
    assert active[:10] == standard[:10] == list(range(99, 89, -1))
    assert active_calls != standard_calls
