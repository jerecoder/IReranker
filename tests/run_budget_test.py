
import sys
from unittest.mock import MagicMock

# Mock missing dependencies
sys.modules["dotenv"] = MagicMock()
sys.modules["loguru"] = MagicMock()
sys.modules["tqdm"] = MagicMock()

from ireranker.oracles import Oracle, BudgetExceeded
from ireranker.rankers import MohajerRanker
from ireranker.types import RankingTask

class MockOracle(Oracle):
    def __init__(self, comparison_limit=None):
        super().__init__(comparison_limit=comparison_limit)
        
    def load_dataset(self, dataset, **kwargs):
        pass
        
    def sample_lt(self, i, j):
        return i > j

def test_oracle_budget_exceeded():
    print("Testing Oracle Budget Exceeded...")
    oracle = MockOracle(comparison_limit=5)
    oracle.set_task(RankingTask(query_id="q1", candidate_ids=["d1", "d2"]))
    
    for _ in range(5):
        oracle.lt(0, 1)
        
    try:
        oracle.lt(0, 1)
        print("FAIL: BudgetExceeded not raised")
        sys.exit(1)
    except BudgetExceeded:
        print("PASS: BudgetExceeded raised")

def test_mohajer_ranker_budget_handling():
    print("Testing MohajerRanker Budget Handling...")
    n = 20
    oracle = MockOracle(comparison_limit=10)
    ranker = MohajerRanker(oracle=oracle, top_k=5)
    
    task = RankingTask(query_id="q1", candidate_ids=[str(i) for i in range(n)])
    
    ranking = ranker.rank(task)
    
    if len(ranking) != n:
        print(f"FAIL: Ranking length mismatch. Expected {n}, got {len(ranking)}")
        sys.exit(1)
        
    if len(set(ranking)) != n:
        print("FAIL: Ranking contains duplicates")
        sys.exit(1)
        
    if oracle.comparisons < 10:
        print(f"FAIL: Comparisons {oracle.comparisons} < 10")
        sys.exit(1)
        
    print("PASS: MohajerRanker handled budget")

def test_mohajer_ranker_no_budget():
    print("Testing MohajerRanker No Budget...")
    n = 10
    oracle = MockOracle(comparison_limit=None)
    ranker = MohajerRanker(oracle=oracle, top_k=5)
    
    task = RankingTask(query_id="q1", candidate_ids=[str(i) for i in range(n)])
    
    ranking = ranker.rank(task)
    
    if len(ranking) != n:
        print(f"FAIL: Ranking length mismatch. Expected {n}, got {len(ranking)}")
        sys.exit(1)
        
    print("PASS: MohajerRanker worked without budget")

if __name__ == "__main__":
    test_oracle_budget_exceeded()
    test_mohajer_ranker_budget_handling()
    test_mohajer_ranker_no_budget()
    print("All tests passed!")
