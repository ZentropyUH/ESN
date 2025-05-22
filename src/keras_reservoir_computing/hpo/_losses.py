# =============================================================
# krc/hpo/_losses.py
# =============================================================
"""Loss function registry.

The module *mirrors* high-level forecasting losses used in reservoir computing.
Any new built-in loss **must** be added to :data:`LOSSES`.
"""
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.stats import gmean
from scipy.special import expit
__all__ = [
    "LossProtocol",
    "LOSSES",
    "get_loss",
    # convenience re-exports
    "forecast_horizon_loss",
    "lyapunov_weighted_loss",
    "multi_step_loss",
]


@runtime_checkable
class LossProtocol(Protocol):
    """Typing protocol for a loss callable."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, /, *args, **kwargs) -> float:  # noqa: D401,E501
        ...


# ------------------------------------------------------------------
# Concrete loss implementations
# ------------------------------------------------------------------

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
        std_y = np.std(y_true, axis=2)
        std_y = np.where(std_y == 0, 1.0, std_y)
        return np.sqrt(np.mean((y_pred - y_true) ** 2, axis=2)) / std_y
    raise ValueError(f"Unknown metric '{metric}'.")


def forecast_horizon_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = "nrmse",
    threshold: float = 0.2,
) -> float:
    """Negative log of valid forecast horizon (to *minimise*).

    This loss penalises the errors with an exponential decay, based on the Lyapunov exponent.
    This is useful for chaotic systems, where the errors grow exponentially.

    Parameters
    ----------
    y_true : np.ndarray
        True values. Shape (B, T, D).
    y_pred : np.ndarray
        Predicted values. Shape (B, T, D).
    metric : str, optional. Default: "nrmse"
        Error metric to compute.
    threshold : float, optional. Default: 0.2
        Threshold for valid forecast horizon.

    Returns
    -------
    float
        Negative log of valid forecast horizon.
    """
    errors = _compute_errors(y_true, y_pred, metric)
    geom_mean = gmean(errors, axis=0)
    valid_len = int(np.sum(geom_mean < threshold))
    return -np.log(valid_len) if valid_len > 0 else np.inf


def lyapunov_weighted_loss(  # noqa: D401,E501
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    dt: float = 1.0,
    lle: float = 1.0,
    metric: str = "rmse",
) -> float:
    """Lyapunov-weighted multi-step geometric mean error.

    This loss penalises the errors with an exponential decay, based on the Lyapunov exponent.
    This is useful for chaotic systems, where the errors grow exponentially.

    Parameters
    ----------
    y_true : np.ndarray
        True values. Shape (B, T, D).
    y_pred : np.ndarray
        Predicted values. Shape (B, T, D).
    dt : float, optional. Default: 1.0
        Time step.
    lle : float, optional. Default: 1.0
        Lyapunov exponent.
    metric : str, optional. Default: "rmse"
        Error metric to compute.

    Returns
    -------
    float
        Lyapunov-weighted multi-step geometric mean error.
    """
    errors = _compute_errors(y_true, y_pred, metric)
    geom_mean = gmean(errors, axis=0)
    timesteps = np.arange(geom_mean.shape[0])
    weights = np.exp(-lle * timesteps * dt)
    return np.sum(weights * geom_mean)


def multi_step_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = "nrmse",
) -> float:
    """
    Sum of the geometric mean error over all timesteps.

    Parameters
    ----------
    y_true : np.ndarray
        True values. Shape (B, T, D).
    y_pred : np.ndarray
        Predicted values. Shape (B, T, D).
    metric : str, optional
        Error metric to compute.

    Returns
    -------
    float
        Sum of the geometric mean error over all timesteps.
    """
    errors = _compute_errors(y_true, y_pred, metric)
    geom_mean = gmean(errors, axis=0)
    return np.sum(geom_mean)


def standard_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = "nrmse",
) -> float:
    """Multi-step geometric mean error (for stable or unstable systems).

    Takes the geometric mean over the batch dimension. Then takes the mean over the timesteps.

    Parameters
    ----------
    y_true : np.ndarray
        True values. Shape (B, T, D).
    y_pred : np.ndarray
        Predicted values. Shape (B, T, D).
    metric : str, optional
        Error metric to compute.

    Returns
    -------
    float
        Mean of the geometric mean error over all timesteps.
    """
    errors = _compute_errors(y_true, y_pred, metric)
    geom_mean = gmean(errors, axis=0)
    return np.mean(geom_mean)


def expected_forecast_horizon(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    metric: str = "rmse",
    threshold: float = 0.2,
    softness: float = 0.02,               # ~ 10 % of threshold is a good default
) -> float:
    """
    Differentiable proxy for forecast-horizon.

    - Rewards models that keep the error below *threshold* for as
      many successive steps as possible.
    - Fully continuous and differentiable, usable in gradient-based optimisation.

    Parameters
    ----------
    y_true, y_pred : (B, T, D) arrays
    metric         : str, error metric passed to `_compute_errors`
    threshold      : float, “acceptable” error
    softness       : float, controls the width of the soft boundary
                     (smaller ⇒ harder threshold)

    Returns
    -------
    float
        Loss to *minimise* (more negative ⇒ longer valid horizon).

    Notes
    -----
    - This loss is equivalent to the expected horizon length, which is the sum of the survival probabilities.
    """
    # 1) per-timestep error, then geometric mean over the batch
    errors = _compute_errors(y_true, y_pred, metric)      # (B, T)
    e_t = gmean(errors, axis=0)                           # (T,)

    # 2) soft indicator of “good prediction” at each step
    good_t = expit((threshold - e_t) / softness)          # ∈ (0, 1)

    # 3) probability that all steps up to t are good (soft horizon survival)
    surv_t = np.exp(np.cumsum(np.log(good_t)))              # (T,)

    # 4) expected horizon length
    H = np.sum(surv_t)

    # 5) loss (minimise → maximise H)
    return -float(H)


LOSSES: dict[str, LossProtocol] = {
    "horizon": forecast_horizon_loss,
    "lyap": lyapunov_weighted_loss,
    "multi_step": multi_step_loss,
    "standard": standard_loss,
    "efh": expected_forecast_horizon,
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


