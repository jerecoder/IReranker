from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RankingTask:
    """A single reranking task for one prompt/query.

    - candidate_ids: identifiers corresponding to the candidate items to rank
    - y_true: optional graded relevance labels aligned with candidate_ids
    """

    query_id: str
    candidate_ids: List[str]
    y_true: Optional[List[float]] = None


@dataclass
class RankingDataset:
    """A collection of tasks to evaluate a ranker."""

    tasks: List[RankingTask]
