"""Functions to train and predict with ESN models."""
# pylint: disable=line-too-long
import json
import os

# ignore tensorflow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from os.path import join

import numpy as np
import pandas as pd

# pylint: disable=no-name-in-module
# pylint: disable=no-member
from keras import initializers
from keras import Model

from src.customs.custom_initializers import (
    ErdosRenyi,
    InputMatrix,
    RegularNX,
    # RegularOwn,
    WattsStrogatzNX,
    # WattsStrogatzOwn,
)
from src.model_instantiators import create_esn_model
from src.forecasters import classic_forecast, section_forecast
from src.plotters import (
    plot_contourf_forecast,
    plot_linear_forecast,
    plot_rmse,
    render_video,
)
from src.readout_generators import linear_readout
from src.utils import load_data, load_model

# pylint: enable=no-name-in-module


def _train(
    # Save params
    data_file: str,
    output_dir: str | None,
    file_name: str | None,
    # General params
    model: str = "ESN",
    units: int = 6000,
    input_initializer: str = "InputMatrix",
    input_bias_initializer: str = "RandomUniform",
    input_scaling: float = 0.5,
    leak_rate: float = 1.0,
    reservoir_activation: str = "tanh",
    seed: int | None = None,
    # Classic Cases
    spectral_radius: float = 0.99,
    reservoir_initializer: str = "WattsStrogatzOwn",
    rewiring: float = 0.5,
    reservoir_degree: int = 3,
    reservoir_sigma: float = 0.5,
    # Parallel cases
    reservoir_amount: int = 10,
    overlap: int = 6,
    # Readout params
    readout_layer: str = "linear",
    regularization: float = 1e-4,
    # Training params
    transient: int = 1000,
    train_length: int = 20000,
):
    """
    Trains an Echo State Network on the data provided in the data file.

    The data file should be a csv file with the rows being the time and the columns being the dimensions.
    The data file should be provided with full path.
    The data file should not include init transient

    Train a model with the data.

    Args:
        data_file (str): Path to the data
        output_dir (str|None): Path to the output file. If None the model will not be saved.
        file_name (str): Name of the file to be saved. If None the model will be saved with the seed name.

    Returns:
        model (ESN_Model): The trained model
        params (dict): The parameters used for training


    """
    ################ GET THE PARAMETERS WITH POSSIBLE RANGES ################

    if seed is None:
        seed = np.random.randint(0, 100000000)

    if file_name is None:
        file_name = f"{seed}"

    params = locals().copy()

    # Only the training data needed
    (
        transient_data,
        train_data,
        train_target,
        _,
        _,
        _,
    ) = load_data(
        data_file,
        transient,
        train_length,
    )

    features = train_data.shape[-1]

    ############### CHOOSE THE INPUT INITIALIZER ###############

    match input_initializer:
        case "InputMatrix":
            input_initializer = InputMatrix(sigma=input_scaling)
        case "RandomUniform":
            input_initializer = initializers.RandomUniform(sigma=input_scaling)

    ############### CHOOSE THE INPUT INITIALIZER ###############

    match input_bias_initializer:
        case "InputMatrix":
            input_bias_initializer = InputMatrix(sigma=input_scaling)
        case "RandomUniform":
            input_bias_initializer = initializers.RandomUniform(
                sigma=input_scaling
            )

        case "None":
            input_bias_initializer = initializers.Zeros()

    ############### CHOOSE THE RESERVOIR INITIALIZER ###############

    match reservoir_initializer:
        case "RegularNX":
            reservoir_initializer = RegularNX(
                degree=reservoir_degree,
                spectral_radius=spectral_radius,
                sigma=reservoir_sigma,
            )
        case "ErdosRenyi":
            reservoir_initializer = ErdosRenyi(
                degree=reservoir_degree,
                spectral_radius=spectral_radius,
                sigma=reservoir_sigma,
            )
        case "WattsStrogatzNX":
            reservoir_initializer = WattsStrogatzNX(
                degree=reservoir_degree,
                spectral_radius=spectral_radius,
                rewiring_p=rewiring,
                sigma=reservoir_sigma,
            )

    ############### CHOOSE THE MODEL ###############

    match model:
        case "ESN":
            _model = create_esn_model(
                features=features,
                units=units,
                leak_rate=leak_rate,
                input_reservoir_init=input_initializer,
                input_bias_init=input_bias_initializer,
                reservoir_kernel_init=reservoir_initializer,
                esn_activation=reservoir_activation,
                seed=seed,
                regularization=regularization,
            )

        case "Parallel-ESN":
            print(f"{model} is yet to be implemented")
            return

        case "Reservoir":
            print(f"{model} is yet to be implemented")
            return

    ############### CHOOSE THE READOUT LAYER ###############

    match readout_layer:
        case "linear":
            _model = linear_readout(
                model=_model,
                transient_data=transient_data,
                train_data=train_data,
                train_target=train_target,
                regularization=regularization,
            )

        case "sgd":
            print("Yet to be implemented")
            return
        case "mlp":
            print("Yet to be implemented")
            return

    ############### SAVING TRAINED MODEL ###############

    if output_dir:
        os.makedirs("Models", exist_ok=True)

        model_name = join(output_dir, file_name)

        # Save the model and save the parameters dictionary in a json file inside the model folder
        _model.save(model_name)

        with open(
            join(model_name, "params.json"),
            "w",
            encoding="utf-8",
        ) as _f_:
            json.dump(params, _f_)

    return (_model, params)


def _forecast(
    trained_model: Model,
    model_params: dict,
    data_file: str,
    output_dir: str,
    forecast_method: str = "classic",
    forecast_length: int = 1000,
    section_initialization_length: int = 50,
    number_of_sections: int = 10,
):
    """Load a model and forecast the data.

    Args:
        trained_model (str): The trained model to be used for forecasting
        model_params (dict): Parameters used for training the model
        data_file (str): The data file to be used for training the model
        output_dir (str): Path for save the forecasted data

        forecast_method (str): The method to be used for forecasting. The default is ClassicForecast.
        forecast_length (int): The number of points to be forecasted. The default is 1000.
        section_initialization_length: int = 50,
        number_of_sections: int = 10,

    Returns:
        None

    """

    transient = model_params["transient"]
    train_length = model_params["train_length"]

    # Load the data
    (
        _,
        _,
        _,
        forecast_transient_data,
        val_data,
        val_target,
    ) = load_data(
        data_file,
        transient=transient,
        train_length=train_length,
    )

    # Load the model

    ############### CHOOSE THE FORECAST METHOD AND FORECAST ###############

    match forecast_method:
        case "classic":
            predictions = classic_forecast(
                trained_model,
                forecast_transient_data,
                val_data,
                val_target,
                forecast_length=forecast_length,
            )
            # this will be of shape (1, forecast_length, features) I need to reshape it to (forecast_length, features)
            predictions = predictions[0]

        case "section":
            predictions = section_forecast(
                trained_model,
                forecast_transient_data,
                val_data,
                val_target,
                section_length=forecast_length,
                section_initialization_length=section_initialization_length,
                number_of_sections=number_of_sections,
            )
            # this will be of shape (1, number_of_sections * forecast_length, features) I need to reshape it to (number_of_sections * forecast_length, features)
            predictions = predictions[0]

    ############### SAVING FORECASTED DATA ###############

    # save in the output directory with the name of the data file (without the path) and the model name attached

    # Prune path from trained_model
    trained_model_name = model_params["file_name"]

    data_name = data_file.split("/")[-1]
    print(data_name)

    if output_dir:
        os.makedirs(f"{output_dir}/{trained_model_name}", exist_ok=True)

        # Save the forecasted data as csv using pandas
        pd.DataFrame(predictions).to_csv(
            f"{output_dir}/{trained_model_name}/{data_name}_{forecast_method}_forecasted.csv",
            index=False,
            header=None,
        )

    else:
        os.makedirs(f"forecasts/{trained_model_name}", exist_ok=True)

        # Save the forecasted data as csv using pandas
        pd.DataFrame(predictions).to_csv(
            f"forecasts/{trained_model_name}/{data_name}_{forecast_method}_forecasted.csv",
            index=False,
            header=None,
        )


def _plot(
    plot_type,
    predictions,
    data_file,
    lyapunov_exponent,
    delta_time,
    plot_points,
    title,
    save_path,
    show,
    y_labels,
    y_values,
    x_label,
    transient,
    train_length,
):
    # Scale time to lyapunov time units
    delta_time = delta_time * lyapunov_exponent

    # Load predictions
    predictions = pd.read_csv(predictions).to_numpy()

    features = predictions.shape[-1]

    # Load the data
    (
        _,
        _,
        _,
        _,
        _,
        val_target,
    ) = load_data(
        data_file,
        transient=transient,
        train_length=train_length,
    )

    # Convert y_labels to a list
    if y_labels:
        y_labels = y_labels.split(",")

    # Convert y_values to a list
    if y_values:
        y_values = [float(i) for i in y_values.split(",")]
        y_values = np.linspace(y_values[0], y_values[1], features)

    # Plot the data
    match plot_type:
        case "linear":
            plot_linear_forecast(
                predictions=predictions,
                val_target=val_target,
                dt=delta_time,
                title=title,
                save_path=save_path,
                show=show,
                ylabels=y_labels,
                xlabel=x_label,
            )

        case "contourf":
            plot_contourf_forecast(
                predictions=predictions,
                val_target=val_target,
                dt=delta_time,
                title=title,
                save_path=save_path,
                show=show,
                xlabel=x_label,
                yvalues=y_values,
            )

        case "rmse":
            plot_rmse(
                predictions=predictions,
                val_target=val_target,
                dt=delta_time,
                title=title,
                save_path=save_path,
                show=show,
                ylabels=y_labels,
                xlabel=x_label,
            )

        case "video":
            render_video(
                predictions=predictions,
                val_target=val_target,
                dt=delta_time,
                title=title,
                save_path=save_path,
                xlabel=x_label,
            )
