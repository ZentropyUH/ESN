import os
from typing import Dict, List, Optional, Union

import keras
from keras.models import clone_model # type: ignore

from tqdm import tqdm

from keras_reservoir_computing.utils.data_utils import list_files_only, load_data
from keras_reservoir_computing.utils.general_utils import timer

from keras_reservoir_computing.utils.model_utils import load_user_config
from keras_reservoir_computing.utils.model_utils.training.training import ReservoirTrainer


# Only works for single readout models. Readout layer name must be "readout". Also the model must not be input-driven.
# Compatible models: ClassicESN, EnsembleESN
def model_trainer(
    model: keras.Model,
    datapath: Union[str, List[str]],
    train_config: Union[str, dict],
    savepath: Optional[str],
    log: bool = False,
    in_place: bool = False,
) -> keras.Model:

    model_copy = model if in_place else clone_model(model)

    # Load training configuration if it's a file path
    if isinstance(train_config, str):
        train_config = load_user_config(train_config)

    init_transient_length = train_config.get("init_transient_length", 5000)
    transient_length = train_config.get("transient_length", 1000)
    train_length = train_config.get("train_length", 20000)
    normalize = train_config.get("normalize", True)

    # Load data
    with timer("Loading data", log=log):
        transient_data, train_data, train_target, _, _, _ = load_data(
            datapath=datapath,
            init_transient=init_transient_length,
            transient=transient_length,
            train_length=train_length,
            normalize=normalize,
        )

    with timer("Warm-up model", log=log):
        model_copy.predict(transient_data, verbose=0)

    # Train the model
    with timer("Training model", log=log):
        model_trainer = ReservoirTrainer(model=model_copy, readout_targets={"readout": train_target})
        model_trainer.fit_readout_layers(warmup_data=transient_data, X=train_data)

    # Save the model
    if savepath is not None:
        with timer("Saving model", log=log):
            directory = os.path.dirname(savepath)
            os.makedirs(name=directory, exist_ok=True)
            if not savepath.endswith(".keras"):
                savepath = savepath + "_model.keras"
            model_copy.save(filepath=savepath)
    return model_copy


def model_batch_trainer(
    model: keras.Model,
    data_folder_path: str,
    train_config: Union[str, Dict],
    savepath: Optional[str] = None,
) -> None:

    data_files = list_files_only(data_folder_path)

    base_savepath = (
        savepath if savepath is not None else os.path.join(data_folder_path, "models")
    )

    for file in tqdm(data_files, desc="Training models"):
        datapath = os.path.join(data_folder_path, file)
        file = file.split(".")[0]
        model_name = f"{model.name}_{file}.keras"
        savepath_file = os.path.join(base_savepath, model_name)

        # Fix: Clone and rebuild model properly
        model_copy = clone_model(model)
        model_copy.build(
            model.input_shape
        )  # Ensure the cloned model has the correct input structure
        model_copy.set_weights(
            model.get_weights()
        )  # Copy weights to prevent unintended behavior

        model_trainer(
            model=model_copy,
            datapath=datapath,
            train_config=train_config,
            savepath=savepath_file,
            in_place=False,
        )
