from typing import List

from .barabasi_albert import barabasi_albert
from .complete import complete
from .connected_erdos_renyi import connected_erdos_renyi
from .connected_watts_strogatz import connected_watts_strogatz
from .dendrocycle import dendrocycle
from .erdos_renyi import erdos_renyi
from .kleinberg_small_world import kleinberg_small_world
from .multi_cycle import multi_cycle
from .newman_watts_strogatz import newman_watts_strogatz
from .regular import regular
from .ring_chord import ring_chord
from .simple_cycle_jumps import simple_cycle_jumps
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
    "multi_cycle",
    "newman_watts_strogatz",
    "regular",
    "ring_chord",
    "simple_cycle_jumps",
    "spectral_cascade",
    "watts_strogatz",
]

def __dir__() -> List[str]:
    return __all__
