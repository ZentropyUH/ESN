from .architectures import classical_ESN, ensemble_with_mean_ESN

__all__ = ["classical_ESN", "ensemble_with_mean_ESN"]


def __dir__() -> list:
    return __all__
