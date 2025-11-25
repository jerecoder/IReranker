import ireranker.rankers.mohajer_ranker  # noqa: F401 - ensure registration side effects
from ireranker.oracles import Oracle
from ireranker.rankers import get_ranker
from ireranker.types import RankingTask


class ScoreOracle(Oracle):
    def __init__(self, scores: list[float]):
        super().__init__()
        self.scores = scores

    def load_dataset(
        self, dataset: str, *, split: str = "test", query_ids=None
    ) -> None:  # noqa: ARG002, ANN001
        return None

    def sample_lt(self, i: int, j: int) -> bool:  # noqa: ARG002
        return self.scores[i] < self.scores[j]


def test_rankers_follow_oracle_pref_order():
    scores = [0.1, 0.9, 0.6, 0.4, 0.8]
    expected = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    task = RankingTask(
        query_id="q0",
        candidate_ids=[f"d{n}" for n in range(len(scores))],
    )

    mohajer = get_ranker("mohajer", oracle=ScoreOracle(scores))
    assert mohajer.rank(task) == expected
