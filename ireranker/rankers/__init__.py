# Ensure built-in rankers are registered on import
from .bubble_ranker import BubbleRanker as BubbleRanker
from .mohajer_ranker import MohajerRanker as MohajerRanker
from .nothing_ranker import NothingRanker as NothingRanker
from .prp_allpairs_ranker import PRPAllpairRanker as PRPAllpairRanker
from .prp_sorting_ranker import PRPSortingRanker as PRPSortingRanker
from .quicksort_ranker import QuicksortTopKRanker as QuicksortTopKRanker
from .ranker import CacheRanker, Ranker, SampleRanker  # noqa: F401
from .registry import (  # noqa: F401
    build_rankers_for_eval,
    default_oracle_for,
    get_ranker,
    list_rankers,
    register_ranker,
)
from .sliding_window import SlidingWindowRanker as SlidingWindowRanker
