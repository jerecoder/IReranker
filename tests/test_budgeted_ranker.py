
import pytest
from ireranker.oracles import Oracle, BudgetExceeded
from ireranker.rankers import MohajerRanker
from ireranker.types import RankingTask

class MockOracle(Oracle):
    def __init__(self, comparison_limit=None):
        super().__init__(comparison_limit=comparison_limit)
        
    def load_dataset(self, dataset, **kwargs):
        pass
        
    def sample_lt(self, i, j):
        # Always say i < j (so i is better than j? No, lt usually means i should be ranked AFTER j)
        # In this codebase, lt(i, j) -> True means i is "less than" j.
        # If we want a consistent ordering 0, 1, 2...
        # If i < j as integers, then i is better?
        # Let's just return i > j for a reverse sorted list, or i < j for sorted.
        return i > j

def test_oracle_budget_exceeded():
    oracle = MockOracle(comparison_limit=5)
    oracle.set_task(RankingTask(query_id="q1", candidate_ids=["d1", "d2"]))
    
    # 5 comparisons allowed
    for _ in range(5):
        oracle.lt(0, 1)
        
    with pytest.raises(BudgetExceeded):
        oracle.lt(0, 1)

def test_mohajer_ranker_budget_handling():
    # Setup a task with enough items to trigger comparisons
    n = 20
    oracle = MockOracle(comparison_limit=10) # Very low limit
    ranker = MohajerRanker(oracle=oracle, top_k=5)
    
    task = RankingTask(query_id="q1", candidate_ids=[str(i) for i in range(n)])
    
    # This should not raise exception, but return a partial ranking
    ranking = ranker.rank(task)
    
    assert len(ranking) == n
    assert len(set(ranking)) == n
    # Check that we actually did some comparisons
    assert oracle.comparisons > 0
    assert oracle.comparisons >= 10 # Should have hit the limit

def test_mohajer_ranker_no_budget():
    n = 10
    oracle = MockOracle(comparison_limit=None)
    ranker = MohajerRanker(oracle=oracle, top_k=5)
    
    task = RankingTask(query_id="q1", candidate_ids=[str(i) for i in range(n)])
    
    ranking = ranker.rank(task)
    
    assert len(ranking) == n
    assert len(set(ranking)) == n
