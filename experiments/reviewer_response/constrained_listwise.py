from __future__ import annotations

import itertools
import time
from typing import Any, Sequence

from experiments.mohajer_hybrid_probe.engine import SharedFlanT5Engine, UsageMeter


def permutation_text(labels: Sequence[int]) -> str:
    return " > ".join(f"[{int(label)}]" for label in labels)


def permutation_token_sequences(tokenizer: Any, labels: Sequence[int]) -> list[list[int]]:
    canonical = tuple(int(label) for label in labels)
    if len(canonical) < 2 or len(set(canonical)) != len(canonical):
        raise ValueError("labels must contain at least two unique identifiers")
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("Tokenizer must define eos_token_id")
    sequences = [
        list(
            tokenizer.encode(
                permutation_text(order),
                add_special_tokens=False,
            )
        )
        + [int(eos)]
        for order in itertools.permutations(canonical)
    ]
    if len({tuple(sequence) for sequence in sequences}) != len(sequences):
        raise ValueError("Distinct permutations collapsed to duplicate token sequences")
    return sequences


def allowed_next_tokens(
    sequences: Sequence[Sequence[int]], prefix: Sequence[int]
) -> list[int]:
    observed = tuple(int(token) for token in prefix)
    allowed = {
        int(sequence[len(observed)])
        for sequence in sequences
        if len(sequence) > len(observed)
        and tuple(int(token) for token in sequence[: len(observed)]) == observed
    }
    if not allowed:
        raise RuntimeError(f"Decoder prefix left the permutation trie: {observed}")
    return sorted(allowed)


def render_constrained_listwise(
    query: str,
    labeled_passages: Sequence[tuple[int, str]],
) -> str:
    body = "\n".join(
        f"[{int(label)}] {str(passage).strip()}" for label, passage in labeled_passages
    )
    labels = ", ".join(f"[{int(label)}]" for label, _ in labeled_passages)
    return (
        f"Rank these passages by relevance to the query: {query}\n{body}\n"
        f"Order all identifiers {labels} from most relevant to least relevant. "
        "Use every identifier exactly once. Return only the ranking.\nRanking:"
    )


class ConstrainedPermutationDecoder:
    """Greedy FLAN-T5 decoding restricted to complete identifier permutations."""

    def __init__(self, engine: SharedFlanT5Engine) -> None:
        self.engine = engine

    def decode(
        self,
        prompt: str,
        *,
        labels: Sequence[int],
        meter: UsageMeter,
        document_count: int,
    ) -> str:
        engine = self.engine
        torch = engine.torch
        tokenizer = engine.tokenizer
        sequences = permutation_token_sequences(tokenizer, labels)
        inputs = tokenizer(
            [str(prompt)],
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        sequence_length = int(inputs["input_ids"].shape[1])
        if sequence_length > engine.encoder_max_tokens:
            raise ValueError(
                f"Rendered prompt is {sequence_length} tokens; encoder limit is "
                f"{engine.encoder_max_tokens}"
            )
        encoder_nonpad = int(inputs["attention_mask"].sum().item())
        padded_slots = int(inputs["input_ids"].numel())
        max_new_tokens = max(len(sequence) for sequence in sequences)
        # Include T5's decoder start token, matching SharedFlanT5Engine accounting.
        meter.ensure_capacity(encoder_nonpad, 1 + max_new_tokens)
        inputs = {key: value.to(engine.device) for key, value in inputs.items()}
        decoder_start = int(engine.model.config.decoder_start_token_id)

        def prefix_allowed_tokens_fn(_: int, decoder_ids: Any) -> list[int]:
            prefix = [int(token) for token in decoder_ids.tolist()]
            if prefix and prefix[0] == decoder_start:
                prefix = prefix[1:]
            return allowed_next_tokens(sequences, prefix)

        if engine.device_type == "cuda":
            torch.cuda.synchronize(engine.device)
        started = time.perf_counter()
        with torch.inference_mode():
            output_ids = engine.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            )
        if engine.device_type == "cuda":
            torch.cuda.synchronize(engine.device)
        elapsed = time.perf_counter() - started

        meter.directional_prompt_instances += 1
        meter.document_instances += int(document_count)
        meter.generation_invocations += 1
        meter.encoder_nonpad_tokens += encoder_nonpad
        meter.encoder_padded_slots += padded_slots
        meter.decoder_tokens += int(output_ids.numel())
        meter.inference_seconds += elapsed
        meter.choice_events += 1
        return str(tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]).strip()
