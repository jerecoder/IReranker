from __future__ import annotations

import re

from experiments.mohajer_hybrid_probe.engine import UsageMeter
from experiments.reviewer_response.active_setwise import (
    run_active_setwise,
    run_standard_setwise_fixed,
    run_standard_setwise_randomized,
)


class _FakeSetwiseEngine:
    def __init__(self) -> None:
        self.max_document_count = 0

    def truncate_query(self, text: str) -> str:
        return text

    def truncate_passage(self, text: str) -> str:
        return text

    def generate(
        self,
        prompts,
        *,
        meter: UsageMeter,
        max_new_tokens: int,
        decoder_prefix: bool,
        document_counts,
    ):
        assert max_new_tokens == 2
        assert decoder_prefix
        outputs = []
        for prompt, document_count in zip(prompts, document_counts):
            self.max_document_count = max(self.max_document_count, document_count)
            matches = re.findall(r'Passage ([A-Z]): "doc(\d+)"', prompt)
            assert len(matches) == document_count
            label, _ = max(matches, key=lambda pair: int(pair[1]))
            outputs.append(f"Passage {label}")
            meter.directional_prompt_instances += 1
            meter.document_instances += document_count
            meter.generation_invocations += 1
            meter.encoder_nonpad_tokens += 10 * document_count
            meter.decoder_tokens += 2
        return outputs


def _row():
    return {
        "dataset": "dbpedia-entity",
        "query_id": "q",
        "query": "query",
        "candidates": [f"d{index}" for index in range(100)],
        "qrels": {},
    }


def _documents():
    return {f"d{index}": f"doc{index}" for index in range(100)}


def test_true_active_setwise_uses_only_multi_document_choice_events() -> None:
    engine = _FakeSetwiseEngine()
    result = run_active_setwise(
        row=_row(),
        documents=_documents(),
        engine=engine,
        seed=42,
        token_budget=100000,
    )
    assert result.ranking[:10] == [f"d{index}" for index in range(99, 89, -1)]
    assert len(result.ranking) == len(set(result.ranking)) == 100
    assert result.meter.logical_comparisons == 0
    assert result.meter.choice_events == result.meter.directional_prompt_instances > 0
    assert result.meter.document_instances >= 2 * result.meter.choice_events
    assert result.stage_b_tokens == 0
    assert result.stage_a_tokens == result.meter.total_model_tokens
    assert engine.max_document_count == 3


def test_active_and_standard_setwise_share_primitive_not_schedule() -> None:
    active_engine = _FakeSetwiseEngine()
    standard_engine = _FakeSetwiseEngine()
    active = run_active_setwise(
        row=_row(),
        documents=_documents(),
        engine=active_engine,
        seed=42,
        token_budget=100000,
    )
    standard = run_standard_setwise_randomized(
        row=_row(),
        documents=_documents(),
        engine=standard_engine,
        seed=42,
        token_budget=100000,
    )
    assert active.ranking[:10] == standard.ranking[:10]
    assert active.meter.logical_comparisons == standard.meter.logical_comparisons == 0
    assert active.meter.choice_events != standard.meter.choice_events
    assert active_engine.max_document_count == standard_engine.max_document_count == 3


def test_fixed_and_randomized_standard_setwise_share_scheduler() -> None:
    fixed_engine = _FakeSetwiseEngine()
    randomized_engine = _FakeSetwiseEngine()
    fixed = run_standard_setwise_fixed(
        row=_row(),
        documents=_documents(),
        engine=fixed_engine,
        seed=42,
        token_budget=100000,
    )
    randomized = run_standard_setwise_randomized(
        row=_row(),
        documents=_documents(),
        engine=randomized_engine,
        seed=42,
        token_budget=100000,
    )
    assert fixed.ranking[:10] == randomized.ranking[:10]
    assert fixed.meter.choice_events == randomized.meter.choice_events
    assert fixed_engine.max_document_count == randomized_engine.max_document_count == 3
