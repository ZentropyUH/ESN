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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates predictions and (optionally) internal states from a trained model and dataset.

    Parameters
    ----------
    model : str or krc.models.ReservoirComputer
        Either the path to the model file or an already instantiated reservoir computing model.
    datapath : str
        Path to the dataset file.
    train_config : str or dict
        Either the path to the dictionary specifying the training configuration or the dictionary itself.
        Must contain the keys 'init_transient_length', 'train_length', 'transient_length', 'normalize' and 'regularization'.
    forecast_config : str or dict
        Either the path to the dictionary specifying the forecast configuration or the dictionary itself.
        Must contain the keys 'forecast_length' and 'internal_states'.
    log : bool, optional
        Whether to log the process. Defaults to True.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray, np.ndarray)
        A tuple containing:
        - val_target : ndarray
            The validation targets.
        - forecast : ndarray
            The model forecasts.
        - states : ndarray
            The internal states, if `internal_states` is True; otherwise, None.

    Notes
    -----
    The dataset is loaded with the same normalization and partitioning used during training.
    """
    # Load the model if they are paths
    if isinstance(train_config, str):
        train_config = load_train_config(train_config)

    if isinstance(forecast_config, str):
        forecast_config = load_forecast_config(forecast_config)
    #########################################

    ### Failsafe for missing keys ###

    # train_config parameters
    init_transient_length = train_config.get("init_transient_length", 5000)
    transient_length = train_config.get("transient_length", 1000)
    train_length = train_config.get("train_length", 20000)
    normalize = train_config.get("normalize", True)

    # forecast_config parameters
    forecast_length = forecast_config.get("forecast_length", 1000)
    internal_states = forecast_config.get("internal_states", True)

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
    progress: Progress = None,
    task: Task = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates predictions for a batch of data files using a single trained model.

    Parameters
    ----------
    model_path : str
        Path to the model file.
    data_folder_path : str
        Path to the folder containing only the data files.
    train_config : str or dict
        Either the path to the dictionary specifying the training configuration or the dictionary itself.
        Must contain the keys 'init_transient_length', 'train_length', 'transient_length', 'normalize' and 'regularization'.
    forecast_config : str or dict
        Either the path to the dictionary specifying the forecast configuration or the dictionary itself.
        Must contain the keys 'forecast_length' and 'internal_states'.
    savepath : str, optional
        Path to the folder where the predictions and targets will be saved. If None,
        the results will not be saved.
    format : str, optional
        File format to use when saving data. Defaults to "npy".
    log : bool, optional
        Whether to log the process. Defaults to True.
    progress : rich.progress.Progress, optional
        A Rich Progress object for manual progress control.
    task : rich.progress.Task, optional
        A specific Rich Task object to update during the prediction loop.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        A tuple containing:
        - predictions_array : ndarray
            Concatenated forecasts for all data files. Shape will be (N, T, D).
        - targets_array : ndarray
            Concatenated validation targets corresponding to each forecast. Shape will be (N, T, D).

    Notes
    -----
    If both predictions and targets for the given model are found in `savepath` with the
    specified `format`, the prediction step is skipped.

    The predictions and targets are saved as separate files with the model name as a prefix.
    """
    # Load the model if they are paths
    # if isinstance(train_config, str):
    #     train_config = load_train_config(train_config)

    # if isinstance(forecast_config, str):
    #     forecast_config = load_forecast_config(forecast_config)
    #########################################

    data_files = list_files_only(data_folder_path)

    # Initialize empty arrays for concatenation
    predictions_array = None
    targets_array = None

    # Verify if already calculated and saved. If so, skip and notify.
    if savepath is not None:
        # TODO: Change these so I can make modelpath an instance or a path
        pred_filename = model_path.split(".")[0].split("/")[-1] + "_predictions"
        target_filename = model_path.split(".")[0].split("/")[-1] + "_targets"

        exist_predictons = os.path.exists(
            os.path.join(savepath, pred_filename + "." + format)
        )
        exist_targets = os.path.exists(
            os.path.join(savepath, target_filename + "." + format)
        )

        if exist_predictons and exist_targets:

            if progress is not None and task is not None:
                progress.update(task, advance=1)

            print(
                f"Predictions and targets already calculated and saved for model {model_path.split('/')[-1]}. Skipping..."
            )
            return None, None

    # Create progress bar, inner if progress is None, otherwise update task inside the loop
    iterator = (
        track(data_files, description=f"Predicting {model_path.split('/')[-1]}")
        if progress is None
        else data_files
    )

    for data_file in iterator:
        datapath = os.path.join(data_folder_path, data_file)

        val_target, forecast, _ = model_predictor(
            model=model_path,
            datapath=datapath,
            train_config=train_config,
            forecast_config=forecast_config,
            log=log,
        )

        T = min(forecast.shape[1], val_target.shape[1])

        val_target = val_target[:, :T, :]
        forecast = forecast[:, :T, :]

        # Concatenate predictions and targets along the first axis
        if predictions_array is None:
            predictions_array = forecast
        else:
            predictions_array = np.concatenate((predictions_array, forecast), axis=0)

        if targets_array is None:
            targets_array = val_target
        else:
            targets_array = np.concatenate((targets_array, val_target), axis=0)

        # Update global progress bar
        if progress is not None and task is not None:
            progress.update(task, advance=1)

    # Save individual files if savepath is provided
    if savepath is not None:
        os.makedirs(name=savepath, exist_ok=True)

        pred_filename = model_path.split(".")[0].split("/")[-1] + "_predictions"
        save_data(
            data=predictions_array,
            filename=pred_filename,
            savepath=savepath,
            format=format,
        )

        target_filename = model_path.split(".")[0].split("/")[-1] + "_targets"
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
    Generates predictions for multiple models across a batch of data files.

    Parameters
    ----------
    model_folder_path : str
        Path to the folder containing only the model files.
    data_folder_path : str
        Path to the folder containing only the data files.
    train_config : str or dict
        Either the path to the dictionary specifying the training configuration or the dictionary itself.
        Must contain the keys 'init_transient_length', 'train_length', 'transient_length', 'normalize' and 'regularization'.
    forecast_config : str or dict
        Either the path to the dictionary specifying the forecast configuration or the dictionary itself.
        Must contain the keys 'forecast_length' and 'internal_states'.
    savepath : str, optional
        Path to the folder where the predictions and targets will be saved. If None,
        the results will not be saved.
    format : str, optional
        File format to use when saving data. Defaults to "npy".
    log : bool, optional
        Whether to log the process. Defaults to True.

    Notes
    -----
    This function iterates over all models in `model_folder_path` and applies each one to
    all data files in `data_folder_path`, optionally saving the predictions and targets.
    """
    # Load the model if they are paths
    # if isinstance(train_config, str):
    #     train_config = load_train_config(train_config)

    # if isinstance(forecast_config, str):
    #     forecast_config = load_forecast_config(forecast_config)
    #########################################

    model_files = list_files_only(model_folder_path)

    total_models = len(model_files)
    total_data_files = len(list_files_only(data_folder_path))
    total_iterations = total_models * total_data_files

    with Progress() as progress:
        task = progress.add_task("[cyan]Predicting...", total=total_iterations)

        for i, model_file in enumerate(model_files):

            print(f"Predicting with model {i+1}/{len(model_files)}")

            modelpath = os.path.join(model_folder_path, model_file)

            _, _ = model_batch_predictor(
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
