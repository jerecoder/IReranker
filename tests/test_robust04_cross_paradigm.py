from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math

import numpy as np
import pytest

from experiments.robust04_cross_paradigm.analyze import (
    holm_adjust,
    load_rows,
    mark_pareto,
    sign_flip_p_value,
)
from experiments.robust04_cross_paradigm.common import ndcg_at_k, sha256
from experiments.robust04_cross_paradigm.engine import SharedFlanT5Engine, UsageMeter
from experiments.robust04_cross_paradigm.methods import _apply_permutation
from ireranker.data.public_tasks import parse_robust_topics_from_testset_gz


def test_robust04_multiline_titles_are_parsed() -> None:
    payload = gzip.compress(
        b"""
<top>
<num> Number: 651
<title>
multiline robust topic title
<desc> Description:
description here
</top>
"""
    )
    assert parse_robust_topics_from_testset_gz(payload) == [
        {"_id": "651", "text": "multiline robust topic title", "metadata": {}}
    ]


def test_ndcg_uses_full_qrels_for_ideal_ranking() -> None:
    qrels = {"outside": 2, "a": 1, "b": 0}
    assert ndcg_at_k(["a", "b"], qrels, 1) == pytest.approx(1.0 / 2.0)


def test_ndcg_matches_pytrec_eval_linear_gain() -> None:
    pytrec_eval = pytest.importorskip("pytrec_eval")
    qrels = {"outside": 2, "a": 1, "b": 0}
    evaluator = pytrec_eval.RelevanceEvaluator({"q": qrels}, {"ndcg_cut.2"})
    expected = evaluator.evaluate({"q": {"a": 2.0, "b": 1.0}})["q"]["ndcg_cut_2"]
    assert ndcg_at_k(["a", "b"], qrels, 2) == pytest.approx(expected)


def test_listwise_permutation_is_deduplicated_and_completed() -> None:
    assert _apply_permutation(["a", "b", "c", "d"], "[3] > [1] > [3]") == [
        "c",
        "a",
        "b",
        "d",
    ]


class _Scalar:
    def __init__(self, value: int) -> None:
        self.value = value

    def item(self) -> int:
        return self.value


class _Tensor:
    def __init__(self, shape: tuple[int, ...], *, summed: int | None = None) -> None:
        self.shape = shape
        self._summed = summed if summed is not None else math.prod(shape)

    def sum(self) -> _Scalar:
        return _Scalar(self._summed)

    def numel(self) -> int:
        return math.prod(self.shape)

    def to(self, _device: str) -> _Tensor:
        return self

    def repeat(self, rows: int, _columns: int) -> _Tensor:
        return _Tensor((rows, self.shape[1]))


class _Tokenizer:
    def __call__(self, *_args, **_kwargs):
        return {
            "input_ids": _Tensor((1, 3)),
            "attention_mask": _Tensor((1, 3), summed=3),
        }

    def batch_decode(self, _ids, *, skip_special_tokens: bool):
        assert skip_special_tokens
        return ["Passage A"]


class _InferenceMode:
    def __enter__(self):
        return None

    def __exit__(self, *_args):
        return False


class _Torch:
    @staticmethod
    def inference_mode() -> _InferenceMode:
        return _InferenceMode()


class _GenerateOnlyModel:
    def __init__(self) -> None:
        self.generated = False

    def __call__(self, **_kwargs):
        raise AssertionError("generation must not use the model forward pass")

    def generate(self, **_kwargs) -> _Tensor:
        self.generated = True
        return _Tensor((1, 4))


def test_shared_engine_uses_generate_and_accounts_for_tokens() -> None:
    engine = SharedFlanT5Engine.__new__(SharedFlanT5Engine)
    engine.torch = _Torch()
    engine.model = _GenerateOnlyModel()
    engine.tokenizer = _Tokenizer()
    engine.device = "cpu"
    engine.device_type = "cpu"
    engine.encoder_max_tokens = 512
    engine.decoder_prefix = _Tensor((1, 2))
    meter = UsageMeter(token_limit=10)

    outputs = engine.generate(
        ["prompt"],
        meter=meter,
        max_new_tokens=2,
        decoder_prefix=True,
        document_counts=[2],
    )

    assert engine.model.generated
    assert outputs == ["Passage A"]
    assert meter.encoder_nonpad_tokens == 3
    assert meter.decoder_tokens == 4
    assert meter.total_model_tokens == 7
    assert meter.document_instances == 2


def test_sign_flip_and_holm_edge_cases() -> None:
    assert sign_flip_p_value(np.zeros(8), resamples=100) == 1.0
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_token_and_gpu_time_pareto_fronts_are_independent() -> None:
    rows = [
        {"name": "a", "avg_total_model_tokens": 1, "avg_inference_seconds": 3, "ndcg10": 0.5},
        {"name": "b", "avg_total_model_tokens": 2, "avg_inference_seconds": 1, "ndcg10": 0.6},
        {"name": "c", "avg_total_model_tokens": 3, "avg_inference_seconds": 2, "ndcg10": 0.55},
    ]
    mark_pareto(rows)
    assert [(row["pareto_tokens"], row["pareto_gpu_time"]) for row in rows] == [
        (True, False),
        (True, True),
        (False, False),
    ]


def _baseline_row() -> dict[str, str | int | float]:
    return {
        "dataset": "robust04",
        "condition": "bm25",
        "method": "bm25",
        "variant": "top100",
        "token_budget": "",
        "seed": 42,
        "query_id": "301",
        "ndcg10": 0.5,
        "logical_comparisons": 0,
        "choice_events": 0,
        "prompt_instances": 0,
        "generation_invocations": 0,
        "document_instances": 0,
        "avg_documents_per_prompt": 0,
        "encoder_nonpad_tokens": 0,
        "encoder_padded_slots": 0,
        "decoder_tokens": 0,
        "total_model_tokens": 0,
        "inference_seconds": 0,
        "invalid_outputs": 0,
        "inconsistent_outputs": 0,
        "query_wall_seconds": 0.01,
        "peak_gpu_memory_bytes": 0,
    }


def _write_completion_marker(tmp_path, csv_path, qids: list[str]) -> None:
    run_path = tmp_path / "bm25.txt"
    run_path.write_text("301 Q0 DOC 1 1 bm25\n", encoding="utf-8")
    signature = {
        "protocol_version": 1,
        "dataset": "robust04",
        "condition": "bm25",
        "method": "bm25",
        "variant": "top100",
        "token_budget": None,
        "seed": 42,
        "query_count": len(qids),
        "query_ids_sha256": hashlib.sha256("\n".join(qids).encode()).hexdigest(),
    }
    marker = {
        "status": "complete",
        "signature": signature,
        "rows": len(qids),
        "per_query_sha256": sha256(csv_path),
        "run_sha256": sha256(run_path),
    }
    (tmp_path / "bm25.done").write_text(
        json.dumps(marker), encoding="utf-8"
    )


def test_analysis_rejects_missing_schema_and_duplicate_queries(tmp_path) -> None:
    row = _baseline_row()
    path = tmp_path / "bm25.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerows([row, row])
    _write_completion_marker(tmp_path, path, ["301", "301"])
    with pytest.raises(ValueError, match="duplicate query_id"):
        load_rows(
            token_budgets=[], seeds=[], expected_queries=2, per_query_dir=tmp_path,
            runs_dir=tmp_path,
        )

    row.pop("decoder_tokens")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    _write_completion_marker(tmp_path, path, ["301"])
    with pytest.raises(ValueError, match="Missing columns"):
        load_rows(
            token_budgets=[], seeds=[], expected_queries=1, per_query_dir=tmp_path,
            runs_dir=tmp_path,
        )
