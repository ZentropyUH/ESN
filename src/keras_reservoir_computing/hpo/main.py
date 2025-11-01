from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Mapping, MutableMapping, Optional, Sequence

# Don’t import TF at runtime here; only for typing.
if TYPE_CHECKING:
    import tensorflow as tf

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from keras_reservoir_computing.utils.tensorflow import suppress_retracing_during_call

from ._losses import LossProtocol, get_loss
from ._objective import build_objective
from ._utils import make_study_name, monitor_name

__all__ = ["run_hpo"]
logger = logging.getLogger(__name__)


@suppress_retracing_during_call
def run_hpo(
    *,
    model_creator: Callable[..., "tf.keras.Model"],
    search_space: Callable[[optuna.trial.Trial], MutableMapping[str, Any]],
    data_loader: Callable[[optuna.trial.Trial], Mapping[str, Any]],
    n_trials: int,
    loss: LossProtocol | str = "efh",
    loss_params: Optional[dict[str, Any]] = None,
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    sampler: Optional[optuna.samplers.BaseSampler] = None,
    seed: Optional[int] = None,
    validate_shapes: bool = False,
    optuna_kwargs: Optional[Mapping[str, Any]] = None,
    verbosity: int = 1,
    monitor_losses: Optional[Sequence[str | LossProtocol]] = None,
    monitor_params: Optional[Mapping[str, Mapping[str, Any]]] = None,
    clip_value: Optional[float] = None,
    prune_on_clip: bool = False,
) -> optuna.Study:
    """Run an Optuna hyperparameter optimization study for reservoir models.

    This function provides a complete, production-ready HPO pipeline that handles:
    - Model creation and validation
    - Data loading and validation
    - Training with automatic readout fitting
    - Multi-step forecasting evaluation
    - Robust error handling and memory management
    - Progress tracking and logging

    The optimization process uses the ReservoirTrainer for fitting readout layers
    and warmup_forecast for evaluation, with configurable loss functions optimized
    for reservoir computing tasks.

    Parameters
    ----------
    model_creator : Callable[..., tf.keras.Model]
        Function that creates a fresh, uncompiled Keras model for each trial.
        Must accept all hyperparameters from ``search_space`` as keyword arguments.
        Should return a model with at least one ReadOut layer.
    search_space : Callable[[optuna.trial.Trial], MutableMapping[str, Any]]
        Function that defines the hyperparameter search space. Accepts an Optuna
        trial object and returns a dictionary of hyperparameters that will be
        passed to ``model_creator``. Use ``trial.suggest_*`` methods to define
        parameters.
    data_loader : Callable[[optuna.trial.Trial], Mapping[str, Any]]
        Function that loads and returns training/validation data. Must return a
        dictionary with the following keys:

        - ``"transient"``: Warmup data for training phase
        - ``"train"``: Training input data
        - ``"train_target"``: Training targets
        - ``"ftransient"``: Warmup data for forecast phase
        - ``"val"``: Validation input data
        - ``"val_target"``: Validation targets

        Optionally include ``"readout_targets"`` for multi-readout models.
    n_trials : int
        Total number of trials to run. If resuming a study, only the remaining
        trials will be executed.
    loss : LossProtocol | str, default="efh"
        Loss function to optimize. Can be:

        - A string key from :data:`LOSSES`: ``"efh"`` (default, recommended),
          ``"horizon"``, ``"lyap"``, ``"multi_step"``, ``"standard"``
        - A custom callable following :class:`LossProtocol`

        The default ``"efh"`` (expected forecast horizon) is optimized for
        chaotic systems and provides smooth, differentiable optimization.
    loss_params : Optional[dict[str, Any]], default=None
        Additional keyword arguments passed to the loss function. For example:
        ``{"threshold": 0.2, "softness": 0.02}`` for ``"efh"``.
    study_name : Optional[str], default=None
        Name for the Optuna study. If None, auto-generated as
        ``<filename>:<function_name>``.
    storage : Optional[str], default=None
        Optuna storage URL (e.g., ``"sqlite:///study.db"``). If None, uses
        in-memory storage.
    sampler : Optional[optuna.samplers.BaseSampler], default=None
        Optuna sampler for hyperparameter selection. If None, uses TPESampler
        with multivariate optimization enabled.
    seed : Optional[int], default=None
        Random seed for reproducibility.
    validate_shapes : bool, default=False
        Whether to validate data tensor shapes for consistency. Disable if
        using dynamic shapes or batching strategies.
    optuna_kwargs : Optional[Mapping[str, Any]], default=None
        Additional keyword arguments passed to :func:`optuna.create_study`.
    verbosity : int, default=1
        Logging verbosity level:

        - 0: Silent (errors only)
        - 1: Normal (info and warnings)
        - 2: Verbose (debug information)

    monitor_losses : Optional[Sequence[str | LossProtocol]], default=None
        Loss functions to monitor during optimization. If None, no monitoring is done.
        Example: ["efh", "horizon"]
    monitor_params : Optional[Mapping[str, Mapping[str, Any]]], default=None
        Additional keyword arguments passed to the monitor loss functions. If None, no additional parameters are passed.
        Example: {"efh": {"threshold": 0.2, "softness": 0.02}, "horizon": {"threshold": 0.2}}
    clip_value : Optional[float], default=None
        Value to clip the monitor losses to. If None, no clipping is done.
    prune_on_clip : bool, default=False
        Whether to prune trials where any monitor loss is clipped. Otherwise, the trial is penalized by the clipped value.

    Returns
    -------
    optuna.Study
        The completed Optuna study object containing all trial results,
        best parameters, and optimization history.

    Raises
    ------
    ValueError
        If arguments are invalid or incompatible.
    TypeError
        If callbacks have incorrect signatures.

    Examples
    --------
    Basic usage with default EFH loss:

    >>> def model_creator(units, spectral_radius):
    ...     # Build reservoir model
    ...     return model
    >>>
    >>> def search_space(trial):
    ...     return {
    ...         "units": trial.suggest_int("units", 100, 1000, step=100),
    ...         "spectral_radius": trial.suggest_float("spectral_radius", 0.5, 1.5),
    ...     }
    >>>
    >>> def data_loader(trial):
    ...     return {
    ...         "transient": X_trans, "train": X_train, "train_target": y_train,
    ...         "ftransient": X_ftrans, "val": X_val, "val_target": y_val,
    ...     }
    >>>
    >>> study = run_hpo(
    ...     model_creator, search_space,
    ...     n_trials=100, data_loader=data_loader,
    ... )
    >>> print(f"Best params: {study.best_params}")
    >>> print(f"Best value: {study.best_value}")

    Using Lyapunov-weighted loss for chaotic systems:

    >>> study = run_hpo(
    ...     model_creator, search_space,
    ...     n_trials=100, data_loader=data_loader,
    ...     loss="lyap_weighted", loss_params={"lle": 1.2, "dt": 0.1},
    ...     storage="sqlite:///study.db",
    ... )

    Notes
    -----
    - Models should be fresh and uncompiled for each trial
    - The function automatically handles memory cleanup between trials
    - Failed trials return ``penalty_value`` instead of raising exceptions
    - Studies can be resumed by using the same ``study_name`` and ``storage``
    - For best results with chaotic systems, use ``loss="efh"`` or ``loss="lyap"``

    See Also
    --------
    :class:`ReservoirTrainer` : For manual readout training
    :func:`warmup_forecast` : For manual forecasting
    :mod:`keras_reservoir_computing.hpo._losses` : Available loss functions
    """
    # Logging
    if verbosity == 0:
        logging.basicConfig(level=logging.ERROR)
        optuna.logging.set_verbosity(optuna.logging.ERROR)
    elif verbosity == 1:
        logging.basicConfig(level=logging.INFO)
        optuna.logging.set_verbosity(optuna.logging.INFO)
    else:
        logging.basicConfig(level=logging.DEBUG)
        optuna.logging.set_verbosity(optuna.logging.DEBUG)

    # Validate
    if n_trials <= 0:
        raise ValueError(f"n_trials must be positive, got {n_trials}")
    if not callable(model_creator):
        raise TypeError("model_creator must be callable")
    if not callable(search_space):
        raise TypeError("search_space must be callable")
    if not callable(data_loader):
        raise TypeError("data_loader must be callable")

    # Loss
    loss_params = loss_params or {}
    base_loss = get_loss(loss)
    resolved_loss = partial(base_loss, **loss_params) if loss_params else base_loss
    logger.info(f"Using loss function: {loss}")
    if loss_params:
        logger.info(f"Loss parameters: {loss_params}")

    # Monitors
    monitor_specs_resolved: list[tuple[str, LossProtocol]] = []
    if monitor_losses:
        for spec in monitor_losses:
            base = get_loss(spec)
            name = monitor_name(spec, base)
            params = (monitor_params or {}).get(name, {})
            mon = partial(base, **params) if params else base
            monitor_specs_resolved.append((name, mon))

    # Sampler
    if sampler is None:
        sampler = TPESampler(multivariate=True, warn_independent_sampling=False, seed=seed)
        logger.info("Using TPESampler with multivariate optimization")

    # Study name
    if study_name is None:
        study_name = make_study_name(model_creator=model_creator)
        logger.info(f"Auto-generated study name: {study_name}")

    # Study
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=sampler,
        pruner=MedianPruner(n_startup_trials=20, n_warmup_steps=64, interval_steps=64),
        **(optuna_kwargs or {}),
    )
    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if completed_trials > 0:
        logger.info(f"Loaded existing study with {completed_trials} completed trials")
        logger.info(f"Best value so far: {study.best_value:.6f}")

    # Objective
    objective = build_objective(
        model_creator=model_creator,
        search_space=search_space,
        loss_fn=resolved_loss,
        data_loader=data_loader,
        validate_shapes=validate_shapes,
        monitor_specs=monitor_specs_resolved,
        clip_value=clip_value,
        prune_on_clip=prune_on_clip,
    )

    # Optimize
    remaining = max(0, n_trials - completed_trials)
    if remaining > 0:
        logger.info(f"Starting optimization: {remaining} trials remaining")
        try:
            study.optimize(
                objective,
                n_trials=remaining,
                catch=(Exception,),
                show_progress_bar=(verbosity > 0),
            )
        except KeyboardInterrupt:
            logger.warning("Optimization interrupted by user")
        except Exception as exc:
            logger.error(f"Optimization failed: {exc}")
            raise

        logger.info(f"Optimization completed: {len(study.trials)} total trials")
        done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        if done > 0:
            logger.info(f"Best value: {study.best_value:.6f}")
            logger.info(f"Best parameters: {study.best_params}")
    else:
        logger.info(f"All {n_trials} trials already completed")

    return study
