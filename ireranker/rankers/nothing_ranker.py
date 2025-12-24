from __future__ import annotations

import os
from typing import List

from ireranker.oracles import BidirectionalFlanSeq2SeqOracle, Oracle

from .ranker import Ranker
from .registry import register_ranker


@register_ranker(
    "BM25",
    oracle_factories=[
        (
            "flan-live",
            lambda seed: BidirectionalFlanSeq2SeqOracle(
                model_name=os.environ.get("FLAN_MODEL_NAME", "google/flan-t5-xl"),
                quantization=os.environ.get("FLAN_QUANT", "8bit"),
                device=os.environ.get("FLAN_DEVICE", "cuda"),
            ),
        )
    ],
)
class NothingRanker(Ranker):
    def __init__(self, oracle: Oracle, seed: int | None = None):
        super().__init__(oracle, seed)

    def _rank(self) -> List[int]:
        self.task = self.task
        order = list(range(self.n))
        return order
