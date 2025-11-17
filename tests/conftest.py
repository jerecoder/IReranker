from __future__ import annotations

import pytest

from ireranker.types import Oracle


class DummyOracle(Oracle):
    def __init__(self) -> None:
        self.loaded: list[tuple[str, str]] = []

    def load_dataset(self, dataset: str, *, split: str = "test") -> None:
        self.loaded.append((dataset, split))

    def sample_outcome(self, query_id: str, doc_a: str, doc_b: str) -> bool:
        return False


@pytest.fixture
def dummy_oracle() -> DummyOracle:
    return DummyOracle()


@pytest.fixture
def dummy_oracle_factory():
    def factory() -> DummyOracle:
        return DummyOracle()

    return factory
