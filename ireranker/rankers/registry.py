from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple, Type

from ireranker.oracles import BidirectionalMatrixOracle, Oracle

from .ranker import Ranker

_REGISTRY: Dict[str, Type[Ranker]] = {}
_ORACLE_FACTORIES: Dict[str, List[Tuple[str | None, Callable[[int | None], Oracle]]]] = {}


def register_ranker(
    name: str,
    *,
    oracle_factories: Sequence[Tuple[str | None, Callable[[int | None], Oracle]]] | None = None,
) -> Callable[[Type[Ranker]], Type[Ranker]]:
    def decorator(cls: Type[Ranker]) -> Type[Ranker]:
        key = name.lower()
        cls.name = key
        _REGISTRY[key] = cls
        if oracle_factories:
            _ORACLE_FACTORIES[key] = list(oracle_factories)
        return cls

    return decorator


def _factories_for(name: str) -> List[Tuple[str | None, Callable[[int | None], Oracle]]]:
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown ranker: {name}. Available: {list_rankers()}")
    if key in _ORACLE_FACTORIES:
        return _ORACLE_FACTORIES[key]
    return [(None, lambda seed: BidirectionalMatrixOracle())]


def default_oracle_for(name: str, *, seed: int | None = None) -> Oracle:
    label, factory = _factories_for(name)[0]
    oracle = factory(seed)
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


def build_rankers_for_eval(names: Sequence[str], *, seed: int | None = None) -> List[Ranker]:
    rankers: List[Ranker] = []
    for name in names:
        for label, factory in _factories_for(name):
            oracle = factory(seed)
            try:
                oracle.set_seed(seed)
            except AttributeError:
                pass
            r = get_ranker(name, oracle=oracle, seed=seed)
            oracle_label = label or getattr(oracle, "name", None) or ""
            display_name = r.name
            if oracle_label:
                display_name = f"{r.name} [{oracle_label}]"
            r.display_name = display_name
            r.oracle_label = oracle_label
            rankers.append(r)
    return rankers
