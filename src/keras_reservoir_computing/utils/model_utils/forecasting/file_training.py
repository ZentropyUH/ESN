import os
from typing import Dict, List, Optional, Union

import keras
from rich.progress import track

from keras_reservoir_computing.utils.data_utils import list_files_only, load_data
from keras_reservoir_computing.utils.general_utils import timer

from keras_reservoir_computing.utils.model_utils import load_user_config
from keras_reservoir_computing.utils.model_utils.training import ReservoirTrainer


# Only works for single readout models. Readout layer name must be "readout"
def model_trainer(
    model: keras.Model,
    datapath: Union[str, List[str]],
    train_config: Union[str, dict],
    savepath: Optional[str],
    log: bool = False,
) -> keras.Model:

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
        model.predict(transient_data, verbose=0)


    # Train the model
    with timer("Training model", log=log):
        model_trainer = ReservoirTrainer(model=model, readout_targets={"readout": train_target})
        model_trainer.fit_readout_layers(X=train_data)

    # Save the model
    if savepath is not None:
        with timer("Saving model", log=log):
            os.makedirs(name=savepath, exist_ok=True)
            fullpath = os.path.join(savepath, f"{model.name}.keras")
            model.save(filepath=fullpath)

    return model


def model_batch_trainer(
    model: keras.Model,
    data_folder_path: str,
    train_config: Union[str, Dict],
    batch_size: int=1,
    savepath: Optional[str] = None,
) -> None:

    data_files = list_files_only(data_folder_path)

    # Group data files into batches, eliminate the last batch if it's smaller than the batch size
    data_batches = [data_files[i:i+batch_size] for i in range(0, len(data_files), step=batch_size)]
    data_batches = [batch for batch in data_batches if len(batch) == batch_size]

    if savepath is None:
        savepath = os.path.join(data_folder_path, "models")

    for batch in track(data_batches, description="Training models"):

        datapath = [os.path.join(data_folder_path, file) for file in batch]
        model_name = "_".join([os.path.splitext(batch[i])[0] for i in range(batch_size)]) + ".keras"
        savepath = os.path.join(savepath, model_name)

        model_trainer(
            model=model,
            datapath=datapath,
            train_config=train_config,
            savepath=savepath,
        )
