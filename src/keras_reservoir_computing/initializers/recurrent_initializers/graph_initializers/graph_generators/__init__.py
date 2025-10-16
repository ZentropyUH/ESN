from typing import List

from .barabasi_albert import barabasi_albert
from .complete import complete
from .connected_erdos_renyi import connected_erdos_renyi
from .connected_watts_strogatz import connected_watts_strogatz
from .dendrocycle import dendrocycle
from .erdos_renyi import erdos_renyi
from .kleinberg_small_world import kleinberg_small_world
from .newman_watts_strogatz import newman_watts_strogatz
from .regular import regular
from .spectral_cascade import spectral_cascade
from .watts_strogatz import watts_strogatz

__all__ = [
    "barabasi_albert",
    "complete",
    "connected_erdos_renyi",
    "connected_watts_strogatz",
    "dendrocycle",
    "erdos_renyi",
    "kleinberg_small_world",
    "newman_watts_strogatz",
    "regular",
    "spectral_cascade",
    "watts_strogatz",
]

def __dir__() -> List[str]:
    return __all__
