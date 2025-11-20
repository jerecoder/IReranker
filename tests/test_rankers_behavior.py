import ireranker.rankers.BubbleRanker  # noqa: F401 - ensure registration side effects
import ireranker.rankers.MohajerRanker  # noqa: F401 - ensure registration side effects
from ireranker.rankers import get_ranker
from ireranker.types import Oracle, RankingTask


class ScoreOracle(Oracle):
    def __init__(self, scores: list[float]):
        self.scores = scores

    def load_dataset(self, dataset: str, *, split: str = "test") -> None:  # noqa: ARG002
        return None

    def sample_lt(self, task: RankingTask, i: int, j: int) -> bool:  # noqa: ARG002
        # True when item i is worse (lower score) than item j
        return self.scores[i] < self.scores[j]


def test_rankers_follow_oracle_pref_order():
    scores = [0.1, 0.9, 0.6, 0.4, 0.8]
    expected = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    task = RankingTask(
        query_id="q0",
        candidate_ids=[f"d{n}" for n in range(len(scores))],
    )

    bubbly = get_ranker("bubbly", oracle=ScoreOracle(scores))
    assert bubbly.rank(task) == expected

    mohajer = get_ranker("mohajer", oracle=ScoreOracle(scores))
    assert mohajer.rank(task) == expected
