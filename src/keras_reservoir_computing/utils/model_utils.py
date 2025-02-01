import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import keras
import numpy as np
from rich.progress import track, Progress

import keras_reservoir_computing as krc
from keras_reservoir_computing.utils import timer

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


def model_loader(filepath):
    """Loads a model from a file and returns it.

    Args:
        filepath (str): Path to the model file.

    Returns:
        krc_models.ReservoirComputer: The loaded model.
    """
    model = krc.models.ReservoirComputer.load(filepath)
    return model


def model_generator(name, model_config, features, seed=None):
    """Generates a model from a configuration dictionary and returns it.

    Args:
        name (str): Name of the model.
        model_config (dict): Configuration dictionary for the model.
        features (int): Number of features in the target data.
        seed (int, optional): Random seed for the model. Defaults to None.

    Returns:
        krc_models.ReservoirComputer: The generated model.
    """
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

    cell = krc.reservoirs.ESNCell(
        **model_config["cell"],
        input_initializer=feedback_init,
        input_bias_initializer=feedback_bias_init,
        kernel_initializer=kernel_init,
    )

    reservoir = krc.reservoirs.EchoStateNetwork(reservoir_cell=cell)

    readout_layer = keras.layers.Dense(
        features, activation="linear", name="readout", trainable=False
    )

    model = krc.models.ReservoirComputer(
        reservoir=reservoir, readout=readout_layer, seed=seed, name=name
    )

    return model


def model_trainer(name, datapath, model_config, train_config, savepath=None, log=False):
    """Trains a model with a dataset and returns it. If a savepath is provided, the model is saved there.

    Args:
        name (str): Name of the model.
        datapath (str): Path to the dataset.
        model_config (dict): Configuration dictionary for the model.
        train_config (dict): Configuration dictionary for the training.
        savepath (str, optional): Path to the FOLDER where to save the model. Defaults to None.

    Returns:
        krc_models.ReservoirComputer: The trained model.

    """
    # Verify if already calculated and saved. If so, skip and notify.
    if savepath is not None:
        exist_model = os.path.exists(os.path.join(savepath, name + ".keras"))
        if exist_model:
            print(f"Model {name} already trained and saved. Skipping...")
            return

    train_length = train_config["train_length"]
    transient_length = train_config["transient_length"]
    normalize = train_config["normalize"]
    regularization = train_config["regularization"]

    with timer("Loading data", log=log):
        transient_data, train_data, train_target, _, _, _ = krc.utils.load_data(
            datapath=datapath,
            train_length=train_length,
            transient=transient_length,
            normalize=normalize,
        )

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

    with timer("Saving model", log=log):
        if savepath is not None:
            os.makedirs(name=savepath, exist_ok=True)
            fullpath = os.path.join(savepath, name + ".keras")
            model.save(filepath=fullpath)

    return model


def model_batch_trainer(
    data_folder_path, model_config, train_config, savepath, log=True
):
    """Function to train a batch of models from a folder of data files

    Args:
        data_folder_path (str): Path to the folder containing the data files and only the data files.
        model_config (dict): Dictionary containing the model configuration.
        train_config (dict): Dictionary containing the training configuration.
        savepath (str): Path to the folder where the models will be saved

    Returns:
        None
    """
    data_files = krc.utils.list_files_only(data_folder_path)

    if savepath is None:
        savepath = os.path.join(data_folder_path, "models")

    for data_file in track(data_files):

        model_name = data_file.split(".")[0]  # No need for .keras here

        datapath = os.path.join(data_folder_path, data_file)

        model_trainer(
            name=model_name,
            datapath=datapath,
            model_config=model_config,
            train_config=train_config,
            savepath=savepath,
            log=log,
        )


def model_predictor(modelpath, datapath, train_config, forecast_config, log=True):
    """Takes a model and a dataset and returns the predictions with the metadata.

    Args:
        modelpath (str): Path to the model file.
        datapath (str): Path to the dataset.
        train_config (dict): Configuration dictionary for the training.
        forecast_config (dict): Configuration dictionary for the forecasting.
        log (bool, optional): Whether to log the process. Defaults to True.

    Returns:
        (val_target, forecast, states): Tuple(np.ndarray, np.ndarray, np.ndarray) containing the targets, the predictions and the internal states.
    """
    train_length = train_config["train_length"]
    transient_length = train_config["transient_length"]

    forecast_length = forecast_config["forecast_length"]
    internal_states = forecast_config["internal_states"]

    model = model_loader(modelpath)

    with timer("Loading data", log=log):
        _, _, _, ftransient, val_data, val_target = krc.utils.load_data(
            datapath=datapath,
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
    model_path,
    data_folder_path,
    train_config,
    forecast_config,
    savepath=None,
    format="npy",
    log=True,
    progress=None,
    task=None,
):
    """Function to predict with a batch of models from a folder of data files

    Args:
        model_path (str): Path to the model file.
        data_folder_path (str): Path to the folder containing the data files and only the data files.
        train_config (dict): Dictionary containing the training configuration.
        forecast_config (dict): Dictionary containing the forecasting configuration.
        savepath (str): Path to the folder where the predictions will be saved
        format (str, optional): Format to save the predictions. Defaults to "npy".
        log (bool, optional): Whether to log the process. Defaults to True.
        progress (Progress, optional): Rich Progress object. Defaults to None.
        task (Task, optional): Rich Task object. Defaults to None.

    Returns:
        (predictions_array, targets_array): Tuple(np.ndarray, np.ndarray) containing the predictions and the targets. The shapes are (n_samples, n_timesteps, n_features).
    """
    data_files = krc.utils.list_files_only(data_folder_path)

    # Initialize empty arrays for concatenation
    predictions_array = None
    targets_array = None

    # Verify if already calculated and saved. If so, skip and notify.
    if savepath is not None:
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
    iterator = track(data_files, description=f"Predicting {model_path.split('/')[-1]}") if progress is None else data_files

    for data_file in iterator:
        datapath = os.path.join(data_folder_path, data_file)

        val_target, forecast, _ = model_predictor(
            modelpath=model_path,
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
        krc.utils.save_data(
            data=predictions_array,
            filename=pred_filename,
            savepath=savepath,
            format=format,
        )

        target_filename = model_path.split(".")[0].split("/")[-1] + "_targets"
        krc.utils.save_data(
            data=targets_array,
            filename=target_filename,
            savepath=savepath,
            format=format,
        )

    return predictions_array, targets_array


def models_batch_predictor(
    model_folder_path,
    data_folder_path,
    train_config,
    forecast_config,
    savepath=None,
    format="npy",
    log=True,
):
    """Function to predict with a batch of models from a folder of data files

    Args:
        model_folder_path (str): Path to the folder containing the model files and only the model files.
        data_folder_path (str): Path to the folder containing the data files and only the data files.
        train_config (dict): Dictionary containing the training configuration.
        forecast_config (dict): Dictionary containing the forecasting configuration.
        savepath (str): Path to the folder where the predictions will be saved
        format (str, optional): Format to save the predictions. Defaults to "npy".
        log (bool, optional): Whether to log the process. Defaults to True.

    Returns:
        None
    """
    model_files = krc.utils.list_files_only(model_folder_path)
    
    total_models = len(model_files)
    total_data_files = len(krc.utils.list_files_only(data_folder_path))
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
    trained_models_folder_path, ensemble_name="Reservoir_Ensemble", log=False
):
    """Function to create an ensemble model from a folder of trained models

    Args:
        trained_models_folder_path (str): Path to the folder containing the trained model files and only the trained model files.
        ensemble_name (str): Name of the ensemble model.
        log (bool, optional): Whether to log the process. Defaults to False.

    Returns:
        krc_models.ReservoirEnsemble: The ensemble model.
    """
    model_files = krc.utils.list_files_only(trained_models_folder_path)

    ensemble_models = []

    with timer("Loading models", log=log):
        for model_file in track(
            model_files,
            description="Loading models"
        ):
            model = model_loader(os.path.join(trained_models_folder_path, model_file))
            ensemble_models.append(model)

    ensemble = krc.models.ReservoirEnsemble(reservoir_computers=ensemble_models, name=ensemble_name)

    return ensemble


if __name__ == '__main__':
    model = ensemble_model_creator(
        "/media/elessar/Data/Pincha/TSDynamics/data/discrete/Henon/Models",
        ensemble_name="HenonEnsemble",
    )

    print(model)
