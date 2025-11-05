from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RankingTask:
    """A single reranking task for one prompt/query.

    - candidate_ids: identifiers corresponding to the candidate items to rank
    - y_true: optional graded relevance labels aligned with candidate_ids
    - features: optional feature dictionary that rankers may use
    """

    query_id: str
    candidate_ids: List[str]
    y_true: Optional[List[float]] = None
    features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RankingDataset:
    """A collection of tasks to evaluate a ranker."""

    tasks: List[RankingTask]
