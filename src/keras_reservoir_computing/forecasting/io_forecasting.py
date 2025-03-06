# import os
from typing import Dict, List, Tuple, Union

import keras
import tensorflow as tf

from keras_reservoir_computing.utils.data import (
    # list_files_only,
    load_data,
)
from keras_reservoir_computing.utils.general import timer

from keras_reservoir_computing.layers.config import load_user_config
from .forecasting import forecast


def model_predictor(
    model: keras.Model,
    datapath: Union[str, List[str]],
    train_config: Union[str, dict],
    horizon: int,
    match_target: bool = False,
    log: bool = True,
) -> Tuple[tf.Tensor, tf.Tensor, Dict[str, List[tf.TensorArray]]]:

    # Load the model if they are paths
    if isinstance(train_config, str):
        train_config = load_user_config(train_config)

    # Extract training configuration
    init_transient_length = train_config.get("init_transient_length", 5000)
    transient_length = train_config.get("transient_length", 1000)
    train_length = train_config.get("train_length", 20000)
    normalize = train_config.get("normalize", True)

    with timer("Loading data", log=log):
        _, _, _, ftransient, val_data, val_target = load_data(
            datapath=datapath,
            init_transient=init_transient_length,
            train_length=train_length,
            transient=transient_length,
            normalize=normalize,
        )

    if match_target:
        horizon = val_target.shape[1]

    # Model warm-up
    with timer("Warm-up model", log=log):
        model.predict(ftransient, verbose=0)

    # Forecasting
    with timer("Forecasting", log=log):
        initial_feedback = val_data[:, :1, :]
        predictions, states = forecast(
            model=model,
            initial_feedback=initial_feedback,
            horizon=horizon,
            verbose=0,
        )

    return predictions, val_target, states


def model_batch_predictor(
    model: keras.Model,
    data_folder_path: str,
    batch_size: int,
    train_config: Union[str, dict],
    horizon: int,
    match_target: bool = False,
    log: bool = True,
):
    pass