# =============================================================
# krc/hpo/_losses.py
# =============================================================
"""Loss function registry.

The module *mirrors* high-level forecasting losses used in reservoir computing.
Any new built-in loss **must** be added to :data:`LOSSES`.
"""
from typing import Protocol, runtime_checkable

import numpy as np

__all__ = [
    "LossProtocol",
    "LOSSES",
    "get_loss",
    # convenience re-exports
    "forecast_horizon_loss",
    "lyapunov_weighted_loss",
    "multi_step_loss_geomean",
]


@runtime_checkable
class LossProtocol(Protocol):
    """Typing protocol for a loss callable."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, /, *args, **kwargs) -> float:  # noqa: D401,E501
        ...


# ------------------------------------------------------------------
# Concrete loss implementations
# ------------------------------------------------------------------
EPS = np.finfo(float).eps


def _compute_errors(y_true: np.ndarray, y_pred: np.ndarray, metric: str = "rmse") -> np.ndarray:  # noqa: D401,E501
    """
    Compute per-timestep errors (B, T) for the requested metric.

    Parameters
    ----------
    y_true : np.ndarray
        True values. Shape: (B, T, D).
    y_pred : np.ndarray
        Predicted values. Shape: (B, T, D).
    metric : str, optional
        The metric to use.

    Returns
    -------
    np.ndarray
        Per-timestep errors. Shape: (B, T).
    """
    assert y_true.shape == y_pred.shape, "Shapes must match"
    if metric == "mse":
        return np.mean((y_pred - y_true) ** 2, axis=2)
    if metric == "rmse":
        return np.sqrt(np.mean((y_pred - y_true) ** 2, axis=2))
    if metric == "mae":
        return np.mean(np.abs(y_pred - y_true), axis=2)
    if metric == "nrmse":
        std_y = np.std(y_true, axis=2) + EPS
        return np.sqrt(np.mean((y_pred - y_true) ** 2, axis=2)) / std_y
    raise ValueError(f"Unknown metric '{metric}'.")


def forecast_horizon_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = "nrmse",
    threshold: float = 0.2,
) -> float:
    """Negative log of valid forecast horizon (to *minimise*)."""
    errors = _compute_errors(y_true, y_pred, metric)
    geom_mean = np.exp(np.mean(np.log(errors + EPS), axis=0))
    valid_len = int(np.sum(geom_mean < threshold))
    return -np.log(valid_len + EPS)


def lyapunov_weighted_loss(  # noqa: D401,E501
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    dt: float = 1.0,
    lle: float = 1.0,
    metric: str = "rmse",
) -> float:
    """Lyapunov-weighted multi-step RMSE (geometric mean)."""
    errors = _compute_errors(y_true, y_pred, metric)
    geom_mean = np.exp(np.mean(np.log(errors + EPS), axis=0))
    timesteps = np.arange(geom_mean.shape[0])
    weights = np.exp(-lle * timesteps * dt)
    return float(np.sum(weights * geom_mean))


def multi_step_loss_geomean(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = "nrmse",
) -> float:
    """Plain multi-step geometric mean error (sum over T)."""
    errors = _compute_errors(y_true, y_pred, metric)
    geom_mean = np.exp(np.mean(np.log(errors + EPS), axis=0))
    return float(np.sum(geom_mean))


LOSSES: dict[str, LossProtocol] = {
    "horizon": forecast_horizon_loss,
    "lyap": lyapunov_weighted_loss,
    "multi_step": multi_step_loss_geomean,
}


def get_loss(key_or_callable: str | LossProtocol) -> LossProtocol:
    """Return a loss callable given a key or callable."""
    if isinstance(key_or_callable, str):
        try:
            return LOSSES[key_or_callable]
        except KeyError as e:  # pragma: no cover
            raise KeyError(f"Unknown loss '{key_or_callable}'. Available: {list(LOSSES)}") from e
    if not isinstance(key_or_callable, LossProtocol):  # type: ignore[arg-type]
        raise TypeError("Loss must be a str key or a LossProtocol-compatible callable.")
    return key_or_callable


