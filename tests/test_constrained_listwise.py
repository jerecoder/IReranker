from __future__ import annotations

from experiments.reviewer_response.constrained_listwise import (
    allowed_next_tokens,
    permutation_text,
    permutation_token_sequences,
)


class _CharacterTokenizer:
    eos_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [ord(character) for character in text]


def test_four_labels_produce_24_unique_complete_sequences() -> None:
    sequences = permutation_token_sequences(_CharacterTokenizer(), [1, 2, 3, 4])
    assert len(sequences) == len({tuple(sequence) for sequence in sequences}) == 24
    assert all(sequence[-1] == 0 for sequence in sequences)


def test_permutation_trie_never_leaves_valid_sequences() -> None:
    sequences = permutation_token_sequences(_CharacterTokenizer(), [1, 2, 3, 4])
    target = sequences[13]
    prefix: list[int] = []
    for token in target:
        assert token in allowed_next_tokens(sequences, prefix)
        prefix.append(token)


def test_permutation_text_is_strict_parser_format() -> None:
    assert permutation_text([3, 1, 4, 2]) == "[3] > [1] > [4] > [2]"
