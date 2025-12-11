# Ensure built-in rankers are registered on import
from .bubble_ranker import BubbleRanker as BubbleRanker
from .mohajer_bubble_ranker import MohajerBubbleRanker as MohajerBubbleRanker
from .mohajer_ranker import MohajerRanker as MohajerRanker
from .nothing_ranker import NothingRanker as NothingRanker
from .pac import PAC2RoundRanker as PAC2RoundRanker
from .pac import PACRanker as PACRanker
from .pac_bubble_ranker import PACBubbleRanker as PACBubbleRanker
from .pac_optimized import PACOptimizedRanker as PACOptimizedRanker
from .prp_allpairs_ranker import PRPAllpairRanker as PRPAllpairRanker
from .prp_sorting_ranker import PRPSortingRanker as PRPSortingRanker
from .quicksort_ranker import QuicksortTopKRanker as QuicksortTopKRanker
from .ranker import Ranker  # noqa: F401
from .registry import (  # noqa: F401
    build_rankers_for_eval,
    default_oracle_for,
    get_ranker,
    list_rankers,
    register_ranker,
)
from .sliding_window import SlidingWindowRanker as SlidingWindowRanker
from .spectral_mle_ranker import SpectralMLERanker as SpectralMLERanker
