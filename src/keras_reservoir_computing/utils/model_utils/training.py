import os
from typing import Optional, Union

import numpy as np
from rich.progress import track

from keras_reservoir_computing.models import ReservoirComputer
from keras_reservoir_computing.utils.data_utils import list_files_only, load_data
from keras_reservoir_computing.utils.general_utils import timer

from .config import load_train_config
from .model import create_model


def model_trainer(
    datapath: str,
    model_config: Union[str, dict],
    train_config: Union[str, dict],
    seed: Optional[int] = None,
    model_name: Optional[str] = None,
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

    if isinstance(train_config, str):
        train_config = load_train_config(train_config)

    if model_name is None:
        model_name = datapath.split("/")[-1].split(".")[0]

    # Verify if already calculated and saved. If so, skip and notify.
    if savepath is not None:
        exist_model = os.path.exists(os.path.join(savepath, model_name + ".keras"))
        if exist_model:
            print(f"Model {model_name} already trained and saved. Skipping...")
            return

    init_transient_length = train_config.get("init_transient_length", 5000)
    transient_length = train_config.get("transient_length", 1000)
    train_length = train_config.get("train_length", 20000)
    normalize = train_config.get("normalize", True)
    regularization = train_config.get("regularization", 1e-4)

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
        model = create_model(
            name=model_name,
            model_config=model_config,
            features=features,
            seed=seed,
            log=log,
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
            fullpath = os.path.join(savepath, model_name + ".keras")
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
            model_name=model_name,
            savepath=savepath,
            log=log,
        )
