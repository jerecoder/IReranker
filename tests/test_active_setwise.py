from __future__ import annotations

import re

from experiments.mohajer_hybrid_probe.engine import UsageMeter
from experiments.reviewer_response.active_setwise import (
    run_active_setwise,
    run_standard_setwise_randomized,
)


class _FakeSetwiseEngine:
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
    result = run_active_setwise(
        row=_row(),
        documents=_documents(),
        engine=_FakeSetwiseEngine(),
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


def test_active_and_standard_setwise_share_primitive_not_schedule() -> None:
    active = run_active_setwise(
        row=_row(),
        documents=_documents(),
        engine=_FakeSetwiseEngine(),
        seed=42,
        token_budget=100000,
    )
    standard = run_standard_setwise_randomized(
        row=_row(),
        documents=_documents(),
        engine=_FakeSetwiseEngine(),
        seed=42,
        token_budget=100000,
    )
    assert active.ranking[:10] == standard.ranking[:10]
    assert active.meter.logical_comparisons == standard.meter.logical_comparisons == 0
    assert active.meter.choice_events != standard.meter.choice_events
