import os
from typing import Optional, Union, Dict, Any

import numpy as np
from rich.progress import track

from keras_reservoir_computing.models import ReservoirComputer
from keras_reservoir_computing.utils.data_utils import list_files_only, load_data
from keras_reservoir_computing.utils.general_utils import timer

from .config import load_train_config
from .model import create_model


def model_trainer(
    datapath: str,
    model_config: Union[str, Dict[str, Any]],
    train_config: Union[str, Dict[str, Any]],
    seed: Optional[int] = None,
    model_name: Optional[str] = None,
    savepath: Optional[str] = None,
    log: bool = False,
) -> ReservoirComputer:
    """
    Train a reservoir computing model using a single dataset and configuration.

    The function loads data from ``datapath`` (e.g., CSV, NPZ, etc.), creates
    a reservoir computing model based on ``model_config``, and trains the model
    according to the parameters specified in ``train_config``.

    Parameters
    ----------
    datapath : str
        Path to the dataset file (e.g., .csv, .npz, .npy, or .nc).
    model_config : str or dict
        Either a file path (JSON) or a dictionary specifying the model configuration.
        It must contain the following keys:

        - ``feedback_init``: Dict with "name" and (optionally) "params".
        - ``feedback_bias_init``: Dict for Keras's :class:`RandomUniform` initializer.
        - ``kernel_init``: Dict with "name" and (optionally) "params".
        - ``cell``: Dict with ESNCell parameters (e.g., ``units``, ``alpha``).

    train_config : str or dict
        Either a file path (JSON) or a dictionary specifying the training configuration.
        It should contain:

        - ``init_transient_length`` (int)
        - ``train_length`` (int)
        - ``transient_length`` (int)
        - ``normalize`` (bool)
        - ``regularization`` (float)

    seed : int, optional
        Seed for random number generation. If ``None``, a random seed is used.
    model_name : str, optional
        Name for the model. If ``None``, it is derived from the dataset filename.
    savepath : str, optional
        Directory path where the trained model should be saved (as ``.keras``).
        If ``None``, the trained model is **not** saved.
    log : bool, optional
        Whether to log major training steps. Defaults to False. Logs timing
        information for data loading, model creation, and model training.

    Returns
    -------
    ReservoirComputer
        The trained reservoir computing model.

    Raises
    ------
    FileNotFoundError
        If `datapath` or any file-based config is not found.
    ValueError
        If `model_config` or `train_config` is malformed or missing required keys.
    OSError
        If a file with the same model name already exists in `savepath`.

    Notes
    -----
    - If a file named ``{model_name}.keras`` already exists in ``savepath``, the
      function prints a message and returns immediately without retraining.
    - The initial part of the dataset (``init_transient_length``) is discarded,
      and the next ``train_length`` portion is used for training. A ``transient_length``
      portion of that training data is used internally for reservoir transient states.
    - The trained model is saved as ``{model_name}.keras`` if ``savepath`` is provided.

    Example
    -------
    .. code-block:: python

        trained_model = model_trainer(
            datapath="data/my_dataset.csv",
            model_config="config/model_config.json",
            train_config="config/train_config.json",
            model_name="my_model",
            savepath="trained_models",
            log=True
        )
    """
    # Load training configuration if it's a file path
    if isinstance(train_config, str):
        train_config = load_train_config(train_config)

    # Derive model name from dataset if none provided
    if model_name is None:
        model_name = os.path.splitext(os.path.basename(datapath))[0]

    # Check if model already exists
    if savepath is not None:
        model_filepath = os.path.join(savepath, model_name + ".keras")
        if os.path.exists(model_filepath):
            print(f"Model '{model_name}' already trained and saved. Skipping...")
            return

    # Extract training hyperparameters with defaults
    init_transient_length = train_config.get("init_transient_length", 5000)
    transient_length = train_config.get("transient_length", 1000)
    train_length = train_config.get("train_length", 20000)
    normalize = train_config.get("normalize", True)
    regularization = train_config.get("regularization", 1e-4)

    # Load data
    with timer("Loading data", log=log):
        transient_data, train_data, train_target, _, _, _ = load_data(
            datapath=datapath,
            init_transient=init_transient_length,
            train_length=train_length,
            transient=transient_length,
            normalize=normalize,
        )

    # Determine or randomize the seed
    if seed is None:
        seed = np.random.randint(0, 1000000)

    features = train_target.shape[-1]

    # Create the model
    with timer("Generating model", log=log):
        model = create_model(
            name=model_name,
            model_config=model_config,
            features=features,
            seed=seed,
            log=log,
        )

    # Train the model
    with timer("Training model", log=log):
        model.train(
            inputs=(transient_data, train_data),
            train_target=train_target,
            regularization=regularization,
            log=log,
        )

    # Save the model
    if savepath is not None:
        with timer("Saving model", log=log):
            os.makedirs(name=savepath, exist_ok=True)
            fullpath = os.path.join(savepath, f"{model_name}.keras")
            model.save(filepath=fullpath)

    return model


def model_batch_trainer(
    data_folder_path: str,
    model_config: Union[str, Dict[str, Any]],
    train_config: Union[str, Dict[str, Any]],
    savepath: Optional[str] = None,
    log: bool = True,
) -> None:
    """
    Train reservoir computing models in bulk, one per data file in a given folder.

    The function iterates over every file in ``data_folder_path``, loading
    the data, creating and training a model based on ``model_config`` and
    ``train_config``, then optionally saving the trained model.

    Parameters
    ----------
    data_folder_path : str
        Path to a folder that contains **only** the data files.
    model_config : str or dict
        Either a path to a JSON file or a dictionary for the model configuration.
        Must contain ``feedback_init``, ``feedback_bias_init``, ``kernel_init``,
        and ``cell`` keys. (Same format as in :func:`model_trainer`.)
    train_config : str or dict
        Either a path to a JSON file or a dictionary for the training configuration.
        Must contain ``init_transient_length``, ``train_length``, ``transient_length``,
        ``normalize``, and ``regularization``. (Same format as in :func:`model_trainer`.)
    savepath : str, optional
        Directory where the trained models are saved. Defaults to creating
        a "models" subfolder in ``data_folder_path`` if not specified.
    log : bool, optional
        Whether to log the training process for each file. Default is True.

    Returns
    -------
    None
        This function does not return anything. It produces side effects by
        training models and potentially saving them to disk.

    Raises
    ------
    FileNotFoundError
        If ``data_folder_path`` is not found or is not accessible.
    OSError
        If saving is requested but writing to ``savepath`` fails.

    Notes
    -----
    - Each data file in ``data_folder_path`` is passed to :func:`model_trainer`.
      The model name is derived by dropping the file extension.
    - If a trained model of the same name already exists in ``savepath``, that
      data file is skipped.
    - The function uses :func:`rich.progress.track` to display progress over
      the data files.

    Example
    -------
    .. code-block:: python

        model_batch_trainer(
            data_folder_path="data",
            model_config="configs/model_config.json",
            train_config="configs/train_config.json",
            savepath="trained_models",
            log=True
        )
    """
    data_files = list_files_only(data_folder_path)

    if savepath is None:
        savepath = os.path.join(data_folder_path, "models")

    for data_file in track(data_files, description="Training models"):
        model_name = os.path.splitext(data_file)[0]
        datapath = os.path.join(data_folder_path, data_file)

        model_trainer(
            datapath=datapath,
            model_config=model_config,
            train_config=train_config,
            seed=None,  # let model_trainer pick a random seed for each data file
            model_name=model_name,
            savepath=savepath,
            log=log,
        )
