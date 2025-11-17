from __future__ import annotations

from typing import Callable, Dict, List, Type

from ireranker.types import Oracle

from .base import Ranker

_REGISTRY: Dict[str, Type[Ranker]] = {}


def register_ranker(name: str) -> Callable[[Type[Ranker]], Type[Ranker]]:
    def decorator(cls: Type[Ranker]) -> Type[Ranker]:
        key = name.lower()
        cls.name = key
        _REGISTRY[key] = cls
        return cls

    return decorator


def list_rankers() -> List[str]:
    return sorted(_REGISTRY.keys())


def get_ranker(name: str, *, oracle: Oracle, **params) -> Ranker:
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown ranker: {name}. Available: {list_rankers()}")
    return _REGISTRY[key](oracle=oracle, **params)
