import os

from keras_reservoir_computing.reservoirs import EchoStateNetwork, ESNCell

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
from typing import Optional, Tuple, Union

import keras
import numpy as np
from rich.progress import Progress, Task, track

import keras_reservoir_computing as krc
from keras_reservoir_computing.models import ReservoirComputer, ReservoirEnsemble
from keras_reservoir_computing.utils.data_utils import (
    list_files_only,
    load_data,
    save_data,
)
from keras_reservoir_computing.utils.general_utils import timer

# region: example_dicts
#################### Test Dicts ####################

model_config = {
    "feedback_init": {
        "name": "InputMatrix",
        "params": {"sigma": 1.38, "ones": False},
    },
    "feedback_bias_init": {"minval": -1.38, "maxval": 1.38},
    "kernel_init": {
        "name": "WattsStrogatzNX",
        "params": {
            "degree": 10,
            "spectral_radius": 0.42,
            "rewiring_p": 0.72,
            "sigma": 1.38,
            "ones": True,
        },
    },
    "cell": {"units": 300, "leak_rate": 0.8, "noise_level": 0.0},
}

train_config = {
    "init_transient_length": 1000,
    "train_length": 20000,
    "transient_length": 5000,
    "normalize": True,
    "regularization": 1e-6,
}

forecast_config = {
    "forecast_length": 3000,
    "internal_states": False,
}

#################### Test Dicts ####################
# endregion: example_dicts


def config_loader(filepath: str, keys: Tuple) -> dict:
    """
    Loads a configuration dictionary from a JSON file.

    This is a helper function to load a configuration dictionary from a JSON file, namely model_config, train_config, and forecast_config.

    Parameters
    ----------
    filepath : str
        Path to the JSON configuration file.
    keys : Tuple[str]
        Tuple of keys that the configuration dictionary should contain.
    Returns
    -------
    dict
        The loaded configuration dictionary.
    """
    with open(filepath, "r") as file:
        config = json.load(file)

    all_keys = config.keys()

    for key in keys:
        if key not in all_keys:
            raise KeyError(
                f"Key {key} not found in the configuration file {filepath}. It should contain the following keys: {keys}"
            )

    return config


def model_loader(filepath: str) -> ReservoirComputer:
    """
    Loads a model from a file.

    Parameters
    ----------
    filepath : str
        Path to the model file.

    Returns
    -------
    krc.models.ReservoirComputer
        The loaded reservoir computing model.

    Notes
    -----
    Ensure that the file at `filepath` is a valid model file previously saved via the
    `krc.models.ReservoirComputer.save` method.
    """
    model = ReservoirComputer.load(filepath)
    return model


def model_generator(
    name: str, model_config: Union[str, dict], features: int, seed: Optional[int] = None
) -> ReservoirComputer:
    """
    Generates a reservoir computing model based on the provided configuration.

    Parameters
    ----------
    name : str
        Name of the model.
    model_config : str or dict
        Either the path to the dictionary specifying the model configuration or the dictionary itself.
        Must contain the keys 'feedback_init', 'feedback_bias_init', 'kernel_init', and 'cell'.
    features : int
        Number of output features for the readout layer.
    seed : int, optional
        Random seed for reproducibility. If None, a random seed is generated.

    Returns
    -------
    krc.models.ReservoirComputer
        The newly created reservoir computing model.

    Notes
    -----
    This function sets a seed for all random initializations if `seed` is not None.
    Otherwise, it randomly generates a seed to initialize the model parameters.
    """

    model_config_keys = ["feedback_init", "feedback_bias_init", "kernel_init", "cell"]
    if isinstance(model_config, str):
        model_config = config_loader(model_config, model_config_keys)

    if seed is not None:
        model_config["seed"] = seed
    else:
        seed = np.random.randint(0, 1000000)
        model_config["seed"] = seed

    feedback_init = model_config["feedback_init"]["name"]

    if feedback_init in krc.initializers.__dict__:
        feedback_init = krc.initializers.__dict__[feedback_init](
            **model_config["feedback_init"]["params"], seed=seed
        )

    feedback_bias_init = model_config["feedback_bias_init"]

    feedback_bias_init = keras.initializers.RandomUniform(
        **feedback_bias_init, seed=seed
    )

    kernel_init = model_config["kernel_init"]["name"]

    if kernel_init in krc.initializers.__dict__:
        kernel_init = krc.initializers.__dict__[kernel_init](
            **model_config["kernel_init"]["params"], seed=seed
        )

    cell = ESNCell(
        **model_config["cell"],
        input_initializer=feedback_init,
        input_bias_initializer=feedback_bias_init,
        kernel_initializer=kernel_init,
    )

    reservoir = EchoStateNetwork(reservoir_cell=cell)

    readout_layer = keras.layers.Dense(
        features, activation="linear", name="readout", trainable=False
    )

    model = ReservoirComputer(
        reservoir=reservoir, readout=readout_layer, seed=seed, name=name
    )

    return model


def model_trainer(
    datapath: str,
    model_config: Union[str, dict],
    train_config: Union[str, dict],
    seed: Optional[int] = None,
    name: Optional[str] = None,
    savepath: Optional[str] = None,
    log: bool = False,
) -> ReservoirComputer:
    """
    Trains a reservoir computing model using the given dataset and configuration.

    Parameters
    ----------
    datapath : str
        Path to the dataset file.
    model_config : str or dict
        Either the path to the dictionary specifying the model configuration or the dictionary itself.
        Must contain the keys 'feedback_init', 'feedback_bias_init', 'kernel_init', and 'cell'.
    train_config : str or dict
        Either the path to the dictionary specifying the training configuration or the dictionary itself.
        Must contain the keys 'init_transient_length', 'train_length', 'transient_length', 'normalize' and 'regularization'.
    name : str, optional
        Name of the model. If None, the name will be derived from the dataset filename.
    savepath : str, optional
        Path to the folder where the trained model should be saved. If None, the model
        will not be saved.
    log : bool, optional
        Whether to log the training process timing, like ensuring ESP, state harvest and readout calculation. Defaults to False.

    Returns
    -------
    krc.models.ReservoirComputer
        The trained reservoir computing model.

    Notes
    -----
    If `savepath` is provided and a model with the same name already exists in that folder,
    the training step is skipped.
    """

    # Load the dictionaries if they are paths
    model_config_keys = ["feedback_init", "feedback_bias_init", "kernel_init", "cell"]
    if isinstance(model_config, str):
        model_config = config_loader(model_config, model_config_keys)

    train_config_keys = [
        "init_transient_length",
        "train_length",
        "transient_length",
        "normalize",
        "regularization",
    ]
    if isinstance(train_config, str):
        train_config = config_loader(train_config, train_config_keys)
    #########################################

    if name is None:
        name = datapath.split("/")[-1].split(".")[0]

    # Verify if already calculated and saved. If so, skip and notify.
    if savepath is not None:
        exist_model = os.path.exists(os.path.join(savepath, name + ".keras"))
        if exist_model:
            print(f"Model {name} already trained and saved. Skipping...")
            return

    init_transient_length = train_config["init_transient_length"]
    transient_length = train_config["transient_length"]
    train_length = train_config["train_length"]
    normalize = train_config["normalize"]
    regularization = train_config["regularization"]

    with timer("Loading data", log=log):
        transient_data, train_data, train_target, _, _, _ = load_data(
            datapath=datapath,
            init_transient=init_transient_length,
            train_length=train_length,
            transient=transient_length,
            normalize=normalize,
        )

    if seed is None:
        seed = np.random.randint(0, 1000000)

    features = train_target.shape[-1]

    with timer("Generating model", log=log):
        model = model_generator(
            name=name, model_config=model_config, features=features, seed=seed
        )

    with timer("Training model", log=log):
        model.train(
            inputs=(transient_data, train_data),
            train_target=train_target,
            regularization=regularization,
            log=log,
        )

    if savepath is not None:
        with timer("Saving model", log=log):
            os.makedirs(name=savepath, exist_ok=True)
            fullpath = os.path.join(savepath, name + ".keras")
            model.save(filepath=fullpath)

    return model


def model_batch_trainer(
    data_folder_path: str,
    model_config: Union[str, dict],
    train_config: Union[str, dict],
    savepath: Optional[str] = None,
    log: bool = True,
) -> None:
    """
    Trains multiple reservoir computing models using a folder of data files.

    Parameters
    ----------
    data_folder_path : str
        Path to the folder containing only the data files.
    model_config : str or dict
        Either the path to the dictionary specifying the model configuration or the dictionary itself.
        Must contain the keys 'feedback_init', 'feedback_bias_init', 'kernel_init', and 'cell'.
    train_config : str or dict
        Either the path to the dictionary specifying the training configuration or the dictionary itself.
        Must contain the keys 'init_transient_length', 'train_length', 'transient_length', 'normalize' and 'regularization'.
    savepath : str, optional
        Path to the folder where the trained models will be saved.
    log : bool, optional
        Whether to log the process. Defaults to True. See `model_trainer` for more

    Notes
    -----
    Each file in `data_folder_path` is used to train a separate model whose name is
    derived from the filename. If a trained model with the same name already exists,
    it is skipped.
    """

    model_config_keys = ["feedback_init", "feedback_bias_init", "kernel_init", "cell"]
    if isinstance(model_config, str):
        model_config = config_loader(model_config, model_config_keys)

    train_config_keys = [
        "init_transient_length",
        "train_length",
        "transient_length",
        "normalize",
        "regularization",
    ]
    if isinstance(train_config, str):
        train_config = config_loader(train_config, train_config_keys)

    data_files = list_files_only(data_folder_path)

    if savepath is None:
        savepath = os.path.join(data_folder_path, "models")

    for data_file in track(data_files):

        model_name = data_file.split(".")[0]  # No need for .keras here

        datapath = os.path.join(data_folder_path, data_file)

        model_trainer(
            datapath=datapath,
            model_config=model_config,
            train_config=train_config,
            name=model_name,
            savepath=savepath,
            log=log,
        )


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
    train_config_keys = [
        "init_transient_length",
        "train_length",
        "transient_length",
        "normalize",
        "regularization",
    ]
    if isinstance(train_config, str):
        train_config = config_loader(train_config, train_config_keys)

    forecast_config_keys = ["forecast_length", "internal_states"]
    if isinstance(forecast_config, str):
        forecast_config = config_loader(forecast_config, forecast_config_keys)
    #########################################

    init_transient_length = train_config["init_transient_length"]
    transient_length = train_config["transient_length"]
    train_length = train_config["train_length"]

    forecast_length = forecast_config["forecast_length"]
    internal_states = forecast_config["internal_states"]

    if isinstance(model, str):
        model = model_loader(model)

    with timer("Loading data", log=log):
        _, _, _, ftransient, val_data, val_target = load_data(
            datapath=datapath,
            init_transient=init_transient_length,
            train_length=train_length,
            transient=transient_length,
            normalize=True,
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
    train_config_keys = [
        "init_transient_length",
        "train_length",
        "transient_length",
        "normalize",
        "regularization",
    ]
    if isinstance(train_config, str):
        train_config = config_loader(train_config, train_config_keys)

    forecast_config_keys = ["forecast_length", "internal_states"]
    if isinstance(forecast_config, str):
        forecast_config = config_loader(forecast_config, forecast_config_keys)
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
    train_config_keys = [
        "init_transient_length",
        "train_length",
        "transient_length",
        "normalize",
        "regularization",
    ]
    if isinstance(train_config, str):
        train_config = config_loader(train_config, train_config_keys)

    forecast_config_keys = ["forecast_length", "internal_states"]
    if isinstance(forecast_config, str):
        forecast_config = config_loader(forecast_config, forecast_config_keys)
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


def ensemble_model_creator(
    trained_models_folder_path: str,
    ensemble_name: str = "Reservoir_Ensemble",
    log: bool = False,
) -> ReservoirEnsemble:
    """
    Creates an ensemble model by loading multiple trained reservoir computing models.

    Parameters
    ----------
    trained_models_folder_path : str
        Path to the folder containing only the trained model files.
    ensemble_name : str, optional
        Name of the ensemble model. Defaults to "Reservoir_Ensemble".
    log : bool, optional
        Whether to log the process of loading models. Defaults to False.

    Returns
    -------
    krc.models.ReservoirEnsemble
        An ensemble model composed of all loaded reservoir computing models.

    Notes
    -----
    Each model contained in the ensemble is loaded from the trained model files found in `trained_models_folder_path`.
    """
    model_files = list_files_only(trained_models_folder_path)

    ensemble_models = []

    with timer("Loading models", log=log):
        for model_file in track(model_files, description="Loading models"):
            model = model_loader(os.path.join(trained_models_folder_path, model_file))
            ensemble_models.append(model)

    ensemble = ReservoirEnsemble(
        reservoir_computers=ensemble_models, name=ensemble_name
    )

    return ensemble


__all__ = [
    # model_utils
    "model_loader",
    "model_generator",
    "model_trainer",
    "model_predictor",
    "model_batch_trainer",
    "model_batch_predictor",
    "models_batch_predictor",
    "ensemble_model_creator",
]


def __dir__():
    return __all__
