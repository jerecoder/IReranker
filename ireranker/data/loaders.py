from __future__ import annotations

from typing import List

from ireranker.types import RankingDataset, RankingTask


def load_synthetic_dataset(n_tasks: int = 3, n_candidates: int = 5) -> RankingDataset:
    tasks: List[RankingTask] = []
    for t in range(n_tasks):
        candidate_ids = [f"d{t}-{i}" for i in range(n_candidates)]
        # Simple graded relevance: descending by index to make identity non-ideal
        y_true = [float((n_candidates - i - 1)) for i in range(n_candidates)]
        tasks.append(
            RankingTask(
                query_id=f"q{t}",
                candidate_ids=candidate_ids,
                y_true=y_true,
                features={},
            )
        )
    return RankingDataset(tasks=tasks)


def load_from_external(*, raise_if_missing: bool = True) -> RankingDataset:
    # Intentionally left as a stub to avoid assuming concrete data layout.
    if raise_if_missing:
        raise NotImplementedError(
            "Implement external data loader based on files you place under data/external/."
        )
    return RankingDataset(tasks=[])
