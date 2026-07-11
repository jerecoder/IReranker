from __future__ import annotations

from dataclasses import dataclass
import random
import time
from typing import Any, Iterable

from ireranker.oracles import BudgetExceeded, Oracle
from ireranker.types import RankingTask


PAIRWISE_PROMPT = '''Given a query "{query}", which of the following two passages is more relevant to the query?

Passage A: "{doc1}"

Passage B: "{doc2}"

Output Passage A or Passage B:'''


@dataclass
class UsageMeter:
    token_limit: int | None = None
    logical_comparisons: int = 0
    choice_events: int = 0
    directional_prompt_instances: int = 0
    document_instances: int = 0
    generation_invocations: int = 0
    encoder_nonpad_tokens: int = 0
    encoder_padded_slots: int = 0
    decoder_tokens: int = 0
    inference_seconds: float = 0.0
    invalid_outputs: int = 0
    inconsistent_outputs: int = 0

    @property
    def total_model_tokens(self) -> int:
        return self.encoder_nonpad_tokens + self.decoder_tokens

    def ensure_capacity(self, encoder_tokens: int, decoder_reserve: int) -> None:
        if self.token_limit is None:
            return
        if self.total_model_tokens + encoder_tokens + decoder_reserve > self.token_limit:
            raise BudgetExceeded(f"Token budget {self.token_limit} would be exceeded")


class SharedFlanT5Engine:
    """One corrected FLAN-T5 generation primitive shared by all paradigms."""

    def __init__(
        self,
        model_name: str = "google/flan-t5-large",
        model_revision: str = "0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a",
        device: str = "cuda",
        encoder_max_tokens: int = 768,
        query_tokens: int = 32,
        passage_tokens: int = 100,
    ) -> None:
        import torch
        from transformers import T5ForConditionalGeneration, T5Tokenizer

        self.torch = torch
        self.model_name = model_name
        self.model_revision = model_revision
        self.device = device
        self.device_type = torch.device(device).type
        self.encoder_max_tokens = int(encoder_max_tokens)
        self.query_tokens = int(query_tokens)
        self.passage_tokens = int(passage_tokens)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, revision=model_revision)
        dtype = torch.float16 if self.device_type == "cuda" else torch.float32
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name, revision=model_revision, torch_dtype=dtype
        )
        self.model.to(device)
        self.model.eval()
        self.decoder_prefix = self.tokenizer.encode(
            "<pad> Passage", return_tensors="pt", add_special_tokens=False
        ).to(device)

    def truncate(self, text: str, limit: int) -> str:
        # Robust04 documents can be thousands of tokens long. We only convert the
        # first `limit` pieces back to text; suppress the tokenizer's misleading
        # model-length warning because the untruncated sequence is never encoded.
        tokens = self.tokenizer.tokenize(str(text), verbose=False)[: int(limit)]
        return self.tokenizer.convert_tokens_to_string(tokens)

    def truncate_query(self, text: str) -> str:
        return self.truncate(text, self.query_tokens)

    def truncate_passage(self, text: str) -> str:
        return self.truncate(text, self.passage_tokens)

    def render_pairwise(self, query: str, doc1: str, doc2: str) -> str:
        return PAIRWISE_PROMPT.format(query=query, doc1=doc1, doc2=doc2)

    def generate(
        self,
        prompts: list[str],
        *,
        meter: UsageMeter,
        max_new_tokens: int,
        decoder_prefix: bool,
        document_counts: list[int] | None = None,
    ) -> list[str]:
        torch = self.torch
        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        sequence_length = int(inputs["input_ids"].shape[1])
        if sequence_length > self.encoder_max_tokens:
            raise ValueError(
                f"Rendered prompt is {sequence_length} tokens; shared encoder limit is "
                f"{self.encoder_max_tokens}. Lower passage_tokens."
            )
        encoder_nonpad = int(inputs["attention_mask"].sum().item())
        padded_slots = int(inputs["input_ids"].numel())
        prefix_length = int(self.decoder_prefix.shape[1]) if decoder_prefix else 1
        decoder_reserve = len(prompts) * (prefix_length + int(max_new_tokens))
        meter.ensure_capacity(encoder_nonpad, decoder_reserve)

        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        kwargs: dict[str, Any] = {
            "attention_mask": inputs["attention_mask"],
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": int(max_new_tokens),
        }
        if decoder_prefix:
            kwargs["decoder_input_ids"] = self.decoder_prefix.repeat(len(prompts), 1)

        if self.device_type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            output_ids = self.model.generate(input_ids=inputs["input_ids"], **kwargs)
        if self.device_type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started

        meter.directional_prompt_instances += len(prompts)
        meter.document_instances += sum(document_counts or [0] * len(prompts))
        meter.generation_invocations += 1
        meter.encoder_nonpad_tokens += encoder_nonpad
        meter.encoder_padded_slots += padded_slots
        meter.decoder_tokens += int(output_ids.numel())
        meter.inference_seconds += elapsed
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    @staticmethod
    def parse_pairwise(output: str) -> str | None:
        normalized = " ".join(str(output).strip().split())
        if normalized == "Passage A":
            return "A"
        if normalized == "Passage B":
            return "B"
        return None

    def compare_sampled(
        self,
        query: str,
        doc_i: str,
        doc_j: str,
        *,
        rng: random.Random,
        meter: UsageMeter,
    ) -> bool:
        """Return True iff canonical doc_i loses to doc_j."""
        if rng.random() < 0.5:
            prompt = self.render_pairwise(query, doc_i, doc_j)
            label = self.parse_pairwise(
                self.generate(
                    [prompt], meter=meter, max_new_tokens=2, decoder_prefix=True,
                    document_counts=[2],
                )[0]
            )
            meter.logical_comparisons += 1
            if label is None:
                meter.invalid_outputs += 1
                return False
            return label == "B"

        prompt = self.render_pairwise(query, doc_j, doc_i)
        label = self.parse_pairwise(
            self.generate(
                [prompt], meter=meter, max_new_tokens=2, decoder_prefix=True,
                document_counts=[2],
            )[0]
        )
        meter.logical_comparisons += 1
        if label is None:
            meter.invalid_outputs += 1
            return False
        return label == "A"

    def compare_bidirectional(
        self,
        query: str,
        doc_a: str,
        doc_b: str,
        *,
        meter: UsageMeter,
    ) -> bool:
        """Return True iff doc_a is consistently preferred to doc_b by PRP."""
        prompts = [
            self.render_pairwise(query, doc_a, doc_b),
            self.render_pairwise(query, doc_b, doc_a),
        ]
        outputs = self.generate(
            prompts, meter=meter, max_new_tokens=2, decoder_prefix=True,
            document_counts=[2, 2],
        )
        meter.logical_comparisons += 1
        labels = [self.parse_pairwise(output) for output in outputs]
        if labels != ["A", "B"]:
            invalid_count = sum(label is None for label in labels)
            if invalid_count:
                meter.invalid_outputs += invalid_count
            else:
                meter.inconsistent_outputs += 1
            return False
        return True


class SharedSamplingOracle(Oracle):
    """Mohajer sampling oracle backed by the exact shared PRP prompt primitive."""

    def __init__(
        self,
        *,
        engine: SharedFlanT5Engine,
        queries: dict[str, str],
        documents: dict[str, str],
        seed: int,
        comparison_limit: int | None = None,
        token_limit: int | None = None,
    ) -> None:
        super().__init__(comparison_limit=comparison_limit, comparison_limit_per_task=True)
        self.engine = engine
        self.queries = queries
        self.documents = documents
        self.master_seed = int(seed)
        self.token_limit = token_limit
        self.meter = UsageMeter(token_limit=token_limit)
        self.name = "Shared PRP prompt / Sampling"
        self.enable_cache(False)

    def load_dataset(self, dataset: str, **_: Any) -> None:
        if dataset != "robust04":
            raise ValueError(f"SharedSamplingOracle only supports robust04, got {dataset}")

    def set_seed(self, seed: int | None) -> None:
        super().set_seed(seed)
        self.master_seed = int(seed or 0)

    def set_task(self, task: RankingTask) -> None:
        if self.current_task is task:
            return
        super().set_task(task)
        import hashlib

        digest = hashlib.sha256(f"{self.master_seed}:{task.query_id}".encode()).digest()
        self._rng = random.Random(int.from_bytes(digest[:8], "big"))
        self.meter = UsageMeter(token_limit=self.token_limit)
        self._query_text = self.engine.truncate_query(self.queries[task.query_id])
        self._document_texts = {
            doc_id: self.engine.truncate_passage(self.documents[doc_id])
            for doc_id in task.candidate_ids
        }

    def sample_lt(self, i: int, j: int) -> bool:
        if self.current_task is None:
            raise RuntimeError("No task set")
        doc_i = self.current_task.candidate_ids[i]
        doc_j = self.current_task.candidate_ids[j]
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
