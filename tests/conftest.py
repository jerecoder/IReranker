from __future__ import annotations

import pytest

from ireranker.oracles import Oracle

# Avoid circular import issues from the non-registered random_ranker module during pytest
# collection (only tests use this stub; production code remains unchanged).
import sys
import types

if "ireranker.rankers.random_ranker" not in sys.modules:
    sys.modules["ireranker.rankers.random_ranker"] = types.ModuleType(
        "ireranker.rankers.random_ranker"
    )


class DummyOracle(Oracle):
    def __init__(self) -> None:
        super().__init__()
        self.loaded: list[tuple[str, str]] = []

    def load_dataset(
        self, dataset: str, *, split: str = "test", query_ids=None
    ) -> None:  # noqa: ANN001 - test helper
        self.loaded.append((dataset, split))

    def sample_lt(self, i: int, j: int) -> bool:  # noqa: ARG002 - test helper
        return False


@pytest.fixture
def dummy_oracle() -> DummyOracle:
    return DummyOracle()


@pytest.fixture
def dummy_oracle_factory():
    def factory() -> DummyOracle:
        return DummyOracle()

    return factory
