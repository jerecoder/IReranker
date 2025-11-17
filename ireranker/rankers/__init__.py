# Ensure built-in baselines are registered on import
from . import baselines as _baselines  # noqa: F401
from .base import Ranker  # noqa: F401
from .BubbleRank import BubbleRanker as BubbleRanker
from .registry import get_ranker, list_rankers, register_ranker  # noqa: F401
