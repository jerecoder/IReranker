from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import Any, Sequence

from experiments.mohajer_hybrid_probe.engine import SharedFlanT5Engine, UsageMeter
from experiments.reviewer_response.active_multiway import (
    active_multiway_topk,
    standard_multiway_heapsort_topk,
)
from experiments.robust04_cross_paradigm.methods import SETWISE_LABELS, render_setwise


@dataclass
class ActiveSetwiseResult:
    ranking: list[str]
    meter: UsageMeter
    stage_a_tokens: int
    stage_b_tokens: int


class SetwiseChooser:
    """One Setwise prompt, optionally randomized within each choice event."""

    def __init__(
        self,
        *,
        query: str,
        documents: dict[str, str],
        engine: SharedFlanT5Engine,
        meter: UsageMeter,
        seed: int,
        query_id: str,
        randomized_presentation: bool,
        max_documents: int,
    ) -> None:
        self.query = engine.truncate_query(query)
        self.documents = documents
        self.engine = engine
        self.meter = meter
        self.randomized_presentation = randomized_presentation
        self.max_documents = max_documents
        digest = hashlib.sha256(f"setwise:{seed}:{query_id}".encode()).digest()
        self.rng = random.Random(int.from_bytes(digest[:8], "big"))
        self.text_cache: dict[str, str] = {}

    def _text(self, doc_id: str) -> str:
        if doc_id not in self.text_cache:
            self.text_cache[doc_id] = self.engine.truncate_passage(self.documents[doc_id])
        return self.text_cache[doc_id]

    def __call__(self, candidates: Sequence[str]) -> str:
        presented = [str(value) for value in candidates]
        if (
            len(presented) < 2
            or len(presented) > self.max_documents
            or len(presented) > len(SETWISE_LABELS)
        ):
            raise ValueError(f"Unsupported setwise match size: {len(presented)}")
        fallback_order = {doc_id: index for index, doc_id in enumerate(presented)}
        if self.randomized_presentation:
            self.rng.shuffle(presented)
        prompt = render_setwise(self.query, [self._text(doc_id) for doc_id in presented])
        output = self.engine.generate(
            [prompt],
            meter=self.meter,
            max_new_tokens=2,
            decoder_prefix=True,
            document_counts=[len(presented)],
        )[0]
        self.meter.choice_events += 1
        normalized = str(output).strip().upper()
        label = normalized[-1:] if normalized else ""
        if label not in SETWISE_LABELS[: len(presented)]:
            self.meter.invalid_outputs += 1
            return min(presented, key=fallback_order.__getitem__)
        return presented[SETWISE_LABELS.index(label)]


def _chooser(
    *,
    row: dict[str, Any],
    documents: dict[str, str],
    engine: SharedFlanT5Engine,
    seed: int,
    token_budget: int,
    randomized_presentation: bool,
    max_documents: int,
) -> tuple[SetwiseChooser, UsageMeter]:
    meter = UsageMeter(token_limit=token_budget)
    chooser = SetwiseChooser(
        query=str(row["query"]),
        documents=documents,
        engine=engine,
        meter=meter,
        seed=seed,
        query_id=str(row["query_id"]),
        randomized_presentation=randomized_presentation,
        max_documents=max_documents,
    )
    return chooser, meter


def run_active_setwise(
    *,
    row: dict[str, Any],
    documents: dict[str, str],
    engine: SharedFlanT5Engine,
    seed: int,
    token_budget: int,
    arity: int = 3,
) -> ActiveSetwiseResult:
    candidates = [str(value) for value in row["candidates"]]
    chooser, meter = _chooser(
        row=row,
        documents=documents,
        engine=engine,
        seed=seed,
        token_budget=token_budget,
        randomized_presentation=True,
        max_documents=arity,
    )

    def choose_index(indices: Sequence[int]) -> int:
        doc_ids = [candidates[index] for index in indices]
        winner = chooser(doc_ids)
        return indices[doc_ids.index(winner)]

    order = active_multiway_topk(
        len(candidates),
        top_k=10,
        arity=arity,
        choose_best=choose_index,
    )
    return ActiveSetwiseResult(
        ranking=[candidates[index] for index in order],
        meter=meter,
        stage_a_tokens=meter.total_model_tokens,
        stage_b_tokens=0,
    )


def run_standard_setwise_randomized(
    *,
    row: dict[str, Any],
    documents: dict[str, str],
    engine: SharedFlanT5Engine,
    seed: int,
    token_budget: int,
    arity: int = 3,
) -> ActiveSetwiseResult:
    candidates = [str(value) for value in row["candidates"]]
    chooser, meter = _chooser(
        row=row,
        documents=documents,
        engine=engine,
        seed=seed,
        token_budget=token_budget,
        randomized_presentation=True,
        max_documents=arity,
    )
    ranking = standard_multiway_heapsort_topk(
        candidates,
        top_k=10,
        arity=arity,
        choose_best=chooser,
    )
    return ActiveSetwiseResult(
        ranking=ranking,
        meter=meter,
        stage_a_tokens=meter.total_model_tokens,
        stage_b_tokens=0,
    )


def run_standard_setwise_fixed(
    *,
    row: dict[str, Any],
    documents: dict[str, str],
    engine: SharedFlanT5Engine,
    seed: int,
    token_budget: int,
    arity: int = 3,
) -> ActiveSetwiseResult:
    """Conventional fixed-presentation Setwise using the identical scheduler."""
    candidates = [str(value) for value in row["candidates"]]
    chooser, meter = _chooser(
        row=row,
        documents=documents,
        engine=engine,
        seed=seed,
        token_budget=token_budget,
        randomized_presentation=False,
        max_documents=arity,
    )
    ranking = standard_multiway_heapsort_topk(
        candidates,
        top_k=10,
        arity=arity,
        choose_best=chooser,
    )
    return ActiveSetwiseResult(
        ranking=ranking,
        meter=meter,
        stage_a_tokens=meter.total_model_tokens,
        stage_b_tokens=0,
    )
