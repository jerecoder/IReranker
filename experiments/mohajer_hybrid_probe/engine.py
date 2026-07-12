from __future__ import annotations

import hashlib
import random
from typing import Any

from ireranker.oracles import Oracle
from ireranker.types import RankingTask

from experiments.robust04_cross_paradigm.engine import SharedFlanT5Engine, UsageMeter


class ProbeOracleBase(Oracle):
    def __init__(
        self,
        *,
        engine: SharedFlanT5Engine,
        dataset: str,
        queries: dict[str, str],
        documents: dict[str, str],
        seed: int,
        token_limit: int | None,
    ) -> None:
        super().__init__(comparison_limit=None, comparison_limit_per_task=True)
        self.engine = engine
        self.dataset = dataset
        self.queries = queries
        self.documents = documents
        self.master_seed = int(seed)
        self.token_limit = token_limit
        self.meter = UsageMeter(token_limit=token_limit)
        self.name = str(getattr(self.__class__, "oracle_name", self.__class__.__name__))
        self.enable_cache(False)

    def load_dataset(self, dataset: str, **_: Any) -> None:
        if dataset != self.dataset:
            raise ValueError(f"Oracle configured for {self.dataset}, got {dataset}")

    def set_seed(self, seed: int | None) -> None:
        super().set_seed(seed)
        self.master_seed = int(seed or 0)

    def set_task(self, task: RankingTask) -> None:
        if self.current_task is task:
            return
        super().set_task(task)
        digest = hashlib.sha256(f"{self.master_seed}:{task.query_id}".encode()).digest()
        self._rng = random.Random(int.from_bytes(digest[:8], "big"))
        self.meter = UsageMeter(token_limit=self.token_limit)
        self._query_text = self.engine.truncate_query(self.queries[task.query_id])
        self._document_texts = {
            doc_id: self.engine.truncate_passage(self.documents[doc_id])
            for doc_id in task.candidate_ids
        }

    def _pair(self, i: int, j: int) -> tuple[str, str]:
        if self.current_task is None:
            raise RuntimeError("No task set")
        return self.current_task.candidate_ids[i], self.current_task.candidate_ids[j]


class ProbeSamplingOracle(ProbeOracleBase):
    """One seeded prompt direction per logical pair, matching the paper's randomized oracle."""

    oracle_name = "Shared PRP prompt / Randomized direction"

    def sample_lt(self, i: int, j: int) -> bool:
        doc_i, doc_j = self._pair(i, j)
        invalid_before = self.meter.invalid_outputs
        loses = self.engine.compare_sampled(
            self._query_text,
            self._document_texts[doc_i],
            self._document_texts[doc_j],
            rng=self._rng,
            meter=self.meter,
        )
        if self.meter.invalid_outputs > invalid_before:
            return i > j
        return loses


class ProbeBidirectionalOracle(ProbeOracleBase):
    """Two prompt directions atomically, used for the classical pairwise baseline."""

    oracle_name = "Shared PRP prompt / Bidirectional"

    def sample_lt(self, i: int, j: int) -> bool:
        doc_i, doc_j = self._pair(i, j)
        invalid_before = self.meter.invalid_outputs
        inconsistent_before = self.meter.inconsistent_outputs
        j_preferred = self.engine.compare_bidirectional(
            self._query_text,
            self._document_texts[doc_j],
            self._document_texts[doc_i],
            meter=self.meter,
        )
        if (
            self.meter.invalid_outputs > invalid_before
            or self.meter.inconsistent_outputs > inconsistent_before
        ):
            return i > j
        return j_preferred


__all__ = [
    "ProbeBidirectionalOracle",
    "ProbeSamplingOracle",
    "SharedFlanT5Engine",
    "UsageMeter",
]
