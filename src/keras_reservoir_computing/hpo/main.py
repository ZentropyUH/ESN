from typing import Any, Callable, Mapping, MutableMapping

import optuna
import tensorflow as tf
from optuna.samplers import TPESampler
from functools import partial
from keras_reservoir_computing.utils.tensorflow import suppress_retracing_during_call

from ._losses import LossProtocol, get_loss
from ._objective import build_objective
from ._utils import make_study_name


@suppress_retracing_during_call
def run_hpo(
    model_creator: Callable[..., "tf.keras.Model"],
    search_space: Callable[[optuna.trial.Trial], MutableMapping[str, Any]],
    *,
    n_trials: int,
    data_loader: Callable[[], Mapping[str, Any]],
    trainer: str = "custom",
    loss: LossProtocol | str = "lyap",
    loss_params: dict[str, Any] = {},
    study_name: str | None = None,
    storage: str | None = None,
    sampler: optuna.samplers.BaseSampler | None = None,
    seed: int | None = None,
    optuna_kwargs: Mapping[str, Any] | None = None,
) -> optuna.Study:
    """Run an Optuna study for reservoir-based models.

    Parameters
    ----------
    model_creator
        Callable building a **fresh** un-compiled Keras model for every trial.
        It *must* accept the hyper-parameters returned by ``search_space`` as
        keyword arguments.
    search_space
        Callable that registers trial parameters via ``trial.suggest_*`` and
        returns **exactly** the mapping later forwarded to ``model_creator``.
    n_trials
        Number of trials to run.
    data_loader
        Zero-arg callable returning a mapping with the keys required by the
        chosen trainer (e.g. ``train_data``, ``train_target``…).  Put your own
        dataset splits in here.
    trainer
        ``"custom"`` → use :class:`krc.training.ReservoirTrainer`.
        ``"fit"``    → use ``model.fit`` with a Keras-compatible loss.
    loss
        Either a string key from :data:`LOSSES` or a callable following the
        :class:`LossProtocol` signature.  *Only* Keras-compatible callables are
        accepted when ``trainer == "fit"``.
    loss_params
        Additional keyword arguments to pass to the loss function.
    study_name
        Optional explicit study name.  Defaults to
        ``f"{model_creator.__name__}_{trainer}"``.
    storage
        Optuna storage URL.  Use SQLite if omitted.
    sampler, seed, optuna_kwargs
        Passed straight through to :func:`optuna.create_study`.

    Returns
    -------
    optuna.Study
        The finished study object.
    """

    # ------------------------------------------------------------------
    # Resolve & validate loss
    # ------------------------------------------------------------------
    base_loss = get_loss(loss)
    resolved_loss = partial(base_loss, **loss_params) if loss_params else base_loss
    if trainer == "fit" and not isinstance(resolved_loss, tf.keras.losses.Loss):
        raise ValueError(
            "When trainer='fit' the loss must be a Keras-compatible loss instance."
        )

    # ------------------------------------------------------------------
    # Prepare Optuna study
    # ------------------------------------------------------------------
    if sampler is None:
        # The warn_independent_sampling=False is for spectral radius being > 1- leak_rate
        sampler = TPESampler(multivariate=True, warn_independent_sampling=False, seed=seed) 

    if study_name is None:
        study_name = make_study_name(
            model_creator=model_creator,
            trainer=trainer,
        )

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=sampler,
        **(optuna_kwargs or {}),
    )

    # ------------------------------------------------------------------
    # Build & register objective
    # ------------------------------------------------------------------
    objective = build_objective(
        model_creator=model_creator,
        search_space=search_space,
        trainer=trainer,
        loss_fn=resolved_loss,
        data_loader=data_loader,
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
    )
    return study

