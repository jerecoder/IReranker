from __future__ import annotations

from typing import Callable, Dict, List, Type

from ireranker.oracles import BidirectionalMatrixOracle, Oracle

from .ranker import Ranker

_REGISTRY: Dict[str, Type[Ranker]] = {}
_ORACLE_FACTORIES: Dict[str, Callable[[int | None], Oracle]] = {}


def register_ranker(
    name: str, *, default_oracle_factory: Callable[[int | None], Oracle] | None = None
) -> Callable[[Type[Ranker]], Type[Ranker]]:
    def decorator(cls: Type[Ranker]) -> Type[Ranker]:
        key = name.lower()
        cls.name = key
        _REGISTRY[key] = cls
        if default_oracle_factory is not None:
            _ORACLE_FACTORIES[key] = default_oracle_factory
        return cls

    return decorator


def default_oracle_for(name: str, *, seed: int | None = None) -> Oracle:
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown ranker: {name}. Available: {list_rankers()}")

    factory = _ORACLE_FACTORIES.get(key)
    oracle = factory(seed) if factory is not None else BidirectionalMatrixOracle()
    oracle.set_seed(seed)
    return oracle


def list_rankers() -> List[str]:
    return sorted(_REGISTRY.keys())


def get_ranker(name: str, *, oracle: Oracle | None = None, **params) -> Ranker:
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown ranker: {name}. Available: {list_rankers()}")
    eff_oracle = oracle or default_oracle_for(key, seed=params.get("seed"))
    return _REGISTRY[key](oracle=eff_oracle, **params)
