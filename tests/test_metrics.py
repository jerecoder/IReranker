import math

from ireranker.evaluation import metrics as M


def test_ndcg_ideal_is_one():
    y = [3, 2, 1, 0]
    ideal = [0, 1, 2, 3]  # already sorted descending by relevance
    assert math.isclose(M.ndcg_at_k(y, ideal, k=4), 1.0, rel_tol=1e-9)


def test_precision_and_map_simple():
    # y>0 is relevant
    y = [1, 0, 1, 0]
    predicted = [0, 1, 2, 3]  # two relevant at positions 1 and 3
    p_at_1 = M.precision_at_k(y, predicted, k=1)
    p_at_2 = M.precision_at_k(y, predicted, k=2)
    ap = M.average_precision(y, predicted)
    assert p_at_1 == 1.0
    assert p_at_2 == 0.5
    # AP = (1/1 + 2/3) / 2
    assert math.isclose(ap, (1.0 + 2.0 / 3.0) / 2.0, rel_tol=1e-9)


def test_mrr_first_relevant():
    y = [0, 0, 1, 0]
    predicted = [2, 0, 1, 3]
    assert M.mrr(y, predicted) == 1.0

