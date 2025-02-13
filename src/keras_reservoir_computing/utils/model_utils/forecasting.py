import os
from typing import Optional, Tuple, Union

import numpy as np
from rich.progress import Progress, Task, track

from keras_reservoir_computing.models import ReservoirComputer
from keras_reservoir_computing.utils.data_utils import (
    list_files_only,
    load_data,
    save_data,
)
from keras_reservoir_computing.utils.general_utils import timer

from .config import (
    load_forecast_config,
    load_train_config,
)
from .model import load_model


def model_predictor(
    model: Union[str, ReservoirComputer],
    datapath: str,
    train_config: Union[str, dict],
    forecast_config: Union[str, dict],
    log: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Generate predictions (and optionally internal states) from a trained model and dataset.

    This function loads (or uses) a reservoir computing model and applies it to a dataset
    to produce forecasts. Additionally, it can return internal states if configured to do so.

    Parameters
    ----------
    model : str or ReservoirComputer
        Either a file path to a saved model or an already-instantiated
        ``ReservoirComputer`` object.
    datapath : str
        Path to the dataset file (e.g., CSV, NPZ, NPY, or NC).
    train_config : str or dict
        Either a path to a JSON file containing the training configuration or a dictionary
        with the following keys:

        - ``init_transient_length``
        - ``train_length``
        - ``transient_length``
        - ``normalize``
        - ``regularization``

        If a key is missing, a fallback default is used (e.g., ``train_length=20000``).
    forecast_config : str or dict
        Either a path to a JSON file containing the forecast configuration or a dictionary
        with the following keys:

        - ``forecast_length``
        - ``internal_states``

        If a key is missing, a fallback default is used.
    log : bool, optional
        Whether to log timing info via the ``timer`` context manager. Default is True.

    Returns
    -------
    val_target : np.ndarray
        The validation targets of shape ``(1, T_val, D)``.
    forecast : np.ndarray
        The model forecasts, typically of shape ``(1, T_for, D)`` (may differ if forecast length
        is truncated).
    states : np.ndarray or None
        The internal states if ``internal_states=True``, otherwise ``None``.

    Notes
    -----
    - The dataset is split consistently with the training setup: an initial transient is discarded,
      and the remaining data is split into training and validation sets.
    - If you pass strings for ``train_config`` or ``forecast_config``, they are automatically
      loaded from JSON via :func:`load_train_config` or :func:`load_forecast_config`.
    - If you pass a string for ``model``, it is loaded via :func:`load_model`.
    """
    # Load the model if they are paths
    if isinstance(train_config, str):
        train_config = load_train_config(train_config)

    if isinstance(forecast_config, str):
        forecast_config = load_forecast_config(forecast_config)

    # Extract training configuration
    init_transient_length = train_config.get("init_transient_length", 5000)
    transient_length = train_config.get("transient_length", 1000)
    train_length = train_config.get("train_length", 20000)
    normalize = train_config.get("normalize", True)

    # Extract forecast configuration
    forecast_length = forecast_config.get("forecast_length", 1000)
    internal_states = forecast_config.get("internal_states", True)

    # Load the model if input is a path
    if isinstance(model, str):
        model = load_model(model)

    with timer("Loading data", log=log):
        _, _, _, ftransient, val_data, val_target = load_data(
            datapath=datapath,
            init_transient=init_transient_length,
            train_length=train_length,
            transient=transient_length,
            normalize=normalize,
        )

    with timer("Forecasting", log=log):
        forecast, states = model.forecast(
            forecast_length=forecast_length,
            forecast_transient_data=ftransient,
            val_data=val_data,
            store_states=internal_states,
        )

    return val_target, forecast, states


def model_batch_predictor(
    model_path: str,
    data_folder_path: str,
    train_config: Union[str, dict],
    forecast_config: Union[str, dict],
    savepath: Optional[str] = None,
    format: str = "npy",
    log: bool = True,
    progress: Optional[Progress] = None,
    task: Optional[Task] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Generate predictions for a batch of data files using a single trained model.

    This function iterates over all files in a data folder, uses a single model
    to forecast each file, and then concatenates the predictions and targets
    along the first dimension (i.e., stacking each new file's forecast/target).

    Parameters
    ----------
    model_path : str
        Path to the file containing the trained model (e.g., .h5, .pkl).
    data_folder_path : str
        Path to a directory containing **only** the data files for prediction.
    train_config : str or dict
        Either a path to a JSON file containing training configuration or a dictionary
        with the keys (``init_transient_length``, ``train_length``, ``transient_length``,
        ``normalize``, ``regularization``). Missing keys have fallback defaults.
    forecast_config : str or dict
        Either a path to a JSON file containing forecast configuration or a dictionary
        with the keys (``forecast_length``, ``internal_states``). Missing keys have
        fallback defaults.
    savepath : str, optional
        Directory where the predictions/targets are saved. If ``None``, results
        are **not** saved.
    format : str, optional
        Output file format if saving (e.g., "csv", "npz", "npy", "nc"). Defaults to "npy".
    log : bool, optional
        Whether to log timing info. Defaults to True.
    progress : rich.progress.Progress, optional
        A Rich progress object to manage or display progress externally.
    task : rich.progress.Task, optional
        A specific Rich Task object to update within the provided `progress`.

    Returns
    -------
    predictions_array : np.ndarray or None
        Concatenated model forecasts of shape ``(N, T, D)``, where N is the number
        of data files. If data was never processed (due to skipping), returns None.
    targets_array : np.ndarray or None
        Concatenated validation targets corresponding to ``predictions_array``.
        Shape ``(N, T, D)``. If data was never processed, returns None.

    Notes
    -----
    - If the relevant predictions and targets are already found in ``savepath`` with
      the correct file names and format, the function skips prediction and returns
      ``(None, None)``.
    - The model is loaded for each file by calling :func:`model_predictor`, and the
      resulting predictions/targets are concatenated along axis 0.
    - If ``savepath`` is not None, the final concatenated arrays are saved under:

      .. code-block:: none

          {model_file_stem}_predictions.{format}
          {model_file_stem}_targets.{format}
    """
    data_files = list_files_only(data_folder_path)

    predictions_array = None
    targets_array = None

    # Derive base filenames for saved predictions/targets
    base_name = model_path.split(".")[0].split("/")[-1]
    pred_filename = base_name + "_predictions"
    target_filename = base_name + "_targets"

    # Check if predictions/targets already exist
    if savepath is not None:
        pred_path = os.path.join(savepath, pred_filename + "." + format)
        targ_path = os.path.join(savepath, target_filename + "." + format)
        exist_predictions = os.path.exists(pred_path)
        exist_targets = os.path.exists(targ_path)

        if exist_predictions and exist_targets:
            # Possibly update Rich progress
            if progress is not None and task is not None:
                progress.update(task, advance=1)

            print(
                f"Predictions and targets already exist for model '{base_name}'. "
                f"Skipping batch prediction."
            )
            return None, None

    # Decide how to iterate (Rich track or internal track)
    iterator = (
        track(data_files, description=f"Predicting {base_name}")
        if progress is None
        else data_files
    )

    # Process each data file
    for data_file in iterator:
        datapath = os.path.join(data_folder_path, data_file)

        val_target, forecast, _ = model_predictor(
            model=model_path,
            datapath=datapath,
            train_config=train_config,
            forecast_config=forecast_config,
            log=log,
        )

        # Align forecast/target lengths
        T = min(forecast.shape[1], val_target.shape[1])
        val_target = val_target[:, :T, :]
        forecast = forecast[:, :T, :]

        # Concatenate along axis=0
        if predictions_array is None:
            predictions_array = forecast
        else:
            predictions_array = np.concatenate((predictions_array, forecast), axis=0)

        if targets_array is None:
            targets_array = val_target
        else:
            targets_array = np.concatenate((targets_array, val_target), axis=0)

        # Update Rich progress
        if progress is not None and task is not None:
            progress.update(task, advance=1)

    # Optionally save final arrays
    if savepath is not None:
        os.makedirs(name=savepath, exist_ok=True)

        # Save predictions
        save_data(
            data=predictions_array,
            filename=pred_filename,
            savepath=savepath,
            format=format,
        )
        # Save targets
        save_data(
            data=targets_array,
            filename=target_filename,
            savepath=savepath,
            format=format,
        )

    return predictions_array, targets_array


def models_batch_predictor(
    model_folder_path: str,
    data_folder_path: str,
    train_config: Union[str, dict],
    forecast_config: Union[str, dict],
    savepath: Optional[str] = None,
    format: str = "npy",
    log: bool = True,
) -> None:
    """
    Generate predictions for **all models** in a folder, applied to **all data files** in a data folder.

    For each model file found in ``model_folder_path``, this function calls
    :func:`model_batch_predictor`, which iterates over all data files in
    ``data_folder_path``.

    Parameters
    ----------
    model_folder_path : str
        Path to a folder containing **only** model files (e.g., .h5, .pkl).
    data_folder_path : str
        Path to a folder containing **only** the data files for prediction.
    train_config : str or dict
        Either a path to a JSON file or a dictionary with the training configuration
        keys (``init_transient_length``, ``train_length``, ``transient_length``,
        ``normalize``, ``regularization``).
    forecast_config : str or dict
        Either a path to a JSON file or a dictionary with the forecast configuration
        keys (``forecast_length``, ``internal_states``).
    savepath : str, optional
        If not None, directory where concatenated predictions/targets are saved
        for each model. Defaults to None.
    format : str, optional
        File format to use when saving data. Defaults to "npy".
    log : bool, optional
        Whether to log progress and timing info. Defaults to True.

    Returns
    -------
    None
        This function does not return anything; it iterates over all models
        and performs batch predictions. If ``savepath`` is not None, results
        are saved to files.

    Notes
    -----
    - A :class:`rich.progress.Progress` object is used internally to track progress
      across all models and data files.
    - The total steps are determined by the product of the number of model files
      and the number of data files.
    - If any model's predictions already exist in ``savepath``, those are skipped.
    """
    model_files = list_files_only(model_folder_path)
    total_models = len(model_files)
    total_data_files = len(list_files_only(data_folder_path))
    total_iterations = total_models * total_data_files

    with Progress() as progress:
        task = progress.add_task("[cyan]Predicting...", total=total_iterations)

        for i, model_file in enumerate(model_files):
            print(f"Predicting with model {i+1}/{len(model_files)}")

            modelpath = os.path.join(model_folder_path, model_file)

            # For each model, run batch predictor over all data files
            model_batch_predictor(
                model_path=modelpath,
                data_folder_path=data_folder_path,
                train_config=train_config,
                forecast_config=forecast_config,
                savepath=savepath,
                format=format,
                log=log,
                progress=progress,
                task=task,
            )
