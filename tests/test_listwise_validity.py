from __future__ import annotations

import pytest

from experiments.reviewer_response.listwise_validity import (
    apply_strict_permutation,
    compact_listwise_prompt,
    legacy_repaired_permutation,
    map_labels_to_documents,
    parse_strict_permutation,
)


@pytest.mark.parametrize(
    "output",
    [
        "[3] > [1] > [4] > [2]",
        " [3]>[1] > [4]> [2] ",
    ],
)
def test_strict_listwise_parser_accepts_only_complete_permutations(output: str) -> None:
    parsed = parse_strict_permutation(output, 4)
    assert parsed.valid
    assert parsed.labels == (3, 1, 4, 2)


@pytest.mark.parametrize(
    ("output", "reason"),
    [
        ("[3] > [1] > [2]", "format_or_count"),
        ("[3] > [1] > [1] > [2]", "duplicate_or_out_of_range"),
        ("[3] > [1] > [4] > [5]", "duplicate_or_out_of_range"),
        ("The answer is [3] > [1] > [4] > [2].", "format_or_count"),
        ("3 > 1 > 4 > 2", "format_or_count"),
        ("", "format_or_count"),
        ("Query 2020: [3] > [1] > [4] > [2]", "format_or_count"),
    ],
)
def test_strict_listwise_parser_rejects_malformed_outputs(
    output: str, reason: str
) -> None:
    parsed = parse_strict_permutation(output, 4)
    assert not parsed.valid
    assert parsed.reason == reason


def test_invalid_output_never_silently_becomes_a_strict_ranking() -> None:
    window = ["a", "b", "c", "d"]
    ranking, parsed = apply_strict_permutation(window, "[3] > [1] > [3]")
    assert ranking is None
    assert not parsed.valid


def test_legacy_repair_documents_the_previous_silent_fallback() -> None:
    window = ["a", "b", "c", "d"]
    assert legacy_repaired_permutation(window, "[3] > [1] > [3]") == [
        "c",
        "a",
        "b",
        "d",
    ]


def test_compact_prompt_demands_every_identifier_once() -> None:
    prompt = compact_listwise_prompt("query", ["a", "b", "c", "d"])
    assert "use every identifier once" in prompt
    assert "[1] > [2] > [3] > [4]" in prompt
    assert prompt.endswith("Permutation:")


def test_label_permutations_map_back_to_document_identity() -> None:
    label_map = {3: "a", 1: "b", 4: "c", 2: "d"}
    assert map_labels_to_documents((1, 3, 2, 4), label_map) == (
        "b",
        "a",
        "d",
        "c",
    )
    with pytest.raises(ValueError):
        map_labels_to_documents((1, 3, 3, 4), label_map)
