from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence


@dataclass(frozen=True)
class PermutationParse:
    raw_output: str
    expected_size: int
    labels: tuple[int, ...]
    valid: bool
    reason: str


def parse_strict_permutation(output: str, expected_size: int) -> PermutationParse:
    """Accept only `[i] > [j] > ...` with every expected identifier exactly once."""
    if expected_size <= 0:
        raise ValueError("expected_size must be positive")
    raw = str(output)
    atom = r"\[(\d+)\]"
    pattern = r"\s*" + r"\s*>\s*".join([atom] * expected_size) + r"\s*"
    match = re.fullmatch(pattern, raw)
    if match is None:
        bracketed = tuple(int(value) for value in re.findall(r"\[(\d+)\]", raw))
        return PermutationParse(raw, expected_size, bracketed, False, "format_or_count")
    labels = tuple(int(value) for value in match.groups())
    expected = tuple(range(1, expected_size + 1))
    if tuple(sorted(labels)) != expected:
        return PermutationParse(raw, expected_size, labels, False, "duplicate_or_out_of_range")
    return PermutationParse(raw, expected_size, labels, True, "valid")


def apply_strict_permutation(
    window: Sequence[str], output: str
) -> tuple[list[str] | None, PermutationParse]:
    parsed = parse_strict_permutation(output, len(window))
    if not parsed.valid:
        return None, parsed
    return [str(window[label - 1]) for label in parsed.labels], parsed


def legacy_repaired_permutation(window: Sequence[str], output: str) -> list[str]:
    """Reproduce the old permissive fallback for diagnostics only."""
    response = [int(value) - 1 for value in re.findall(r"\d+", str(output))]
    unique: list[int] = []
    for value in response:
        if 0 <= value < len(window) and value not in unique:
            unique.append(value)
    unique.extend(value for value in range(len(window)) if value not in unique)
    return [str(window[value]) for value in unique]


def compact_listwise_prompt(query: str, passages: Sequence[str]) -> str:
    body = "\n\n".join(
        f"[{index}] {str(passage).strip()}" for index, passage in enumerate(passages, start=1)
    )
    example = " > ".join(f"[{index}]" for index in range(1, len(passages) + 1))
    return (
        f"Query: {query}\n\n{body}\n\n"
        "Rank all passages from most to least relevant. Return exactly one permutation, "
        "use every identifier once, and output no other text.\n"
        f"Required format: {example}\nPermutation:"
    )
