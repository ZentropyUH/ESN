"""Functions to train and predict with ESN models."""
# pylint: disable=line-too-long
import json
import os
from os.path import join

import numpy as np
import pandas as pd

# pylint: disable=no-name-in-module
from keras.initializers import Zeros
from keras.models import load_model
from torch import NoneType
from tqdm import tqdm

from src.customs.custom_initializers import (
    ErdosRenyi,
    InputMatrix,
    RandomUniform,
    RegularNX,
    RegularOwn,
    WattsStrogatzNX,
    WattsStrogatzOwn,
)
from src.customs.custom_models import ESN, ParallelESN
from src.forecasters import classic_forecast, section_forecast
from src.plotters import (
    plot_contourf_forecast,
    plot_linear_forecast,
    plot_rmse,
    render_video,
)
from src.readout_generators import linear_readout
from src.utils import get_range, load_data, load_model_json

# pylint: enable=no-name-in-module


def _train(
    # Save params
    data_file: str,
    output_dir: str = None,
    file_name : str= None,

    # General params
    model: str = 'ESN',
    units: int = 6000,
    input_initializer: str = 'InputMatrix',
    input_bias_initializer: str = 'RandomUniform',
    input_scaling : int = 0.5,
    leak_rate: int = 1.0,
    reservoir_activation: str = 'tanh',

    # Classic Cases
    spectral_radius : int = 0.99,
    reservoir_initializer: str = 'WattsStrogatzOwn',
    rewiring: int = 0.5,
    reservoir_degree: int = 3,
    reservoir_sigma: int = 0.5,

    # Parallel cases
    reservoir_amount: int = 10,
    overlap: int = 6,

    # Readout params
    readout_layer: str = 'linear',
    regularization : int = 1e-4,

    # Training params
    transient: int = 1000,
    train_length : int = 20000,

    # Save flag
    save_model: bool = False
    
):
    """
    Trains an Echo State Network on the data provided in the data file.

    The data file should be a csv file with the rows being the time and the columns being the dimensions. 
    The data file should be provided with full path.
    The data file should not include init transient

    Train a model with the data.

    Args:
        data_file (str): Path to the data
        output_dir (str): Path to the output file
        file_name: Output file name,

    Returns:
        model (ESN_Model): The trained model
        params (dict): The parameters used for training 

     
    """
    ################ GET THE PARAMETERS WITH POSSIBLE RANGES ################

    params = locals().copy()

    # Only the training data needed
    (
        transient_data,
        train_data,
        train_target,
        _,
        _,
        _,

    ) = load_data (
        data_file,
        transient,
        train_length,
    )

    ############### CHOOSE THE INPUT INITIALIZER ###############

    match input_initializer:
        case "InputMatrix":
            input_initializer = (
                InputMatrix(
                    sigma=input_scaling
                )
            )
        case "RandomUniform":
            input_initializer = (
                RandomUniform(
                    sigma=input_scaling
                )
            )

    ############### CHOOSE THE INPUT INITIALIZER ###############

    match input_bias_initializer:
        case "InputMatrix":
            input_bias_initializer = (
                InputMatrix(
                    sigma=input_scaling
                )
            )
        case "RandomUniform":
            input_bias_initializer = (
                RandomUniform(
                    sigma=input_scaling
                )
            )

        case "None":
            input_bias_initializer = (
                Zeros()
            )

    ############### CHOOSE THE RESERVOIR INITIALIZER ###############

    match reservoir_initializer:
        case "RegularOwn":
            reservoir_initializer = RegularOwn(
                degree= reservoir_degree,
                spectral_radius= spectral_radius,
                sigma= reservoir_sigma,
            )
        case "RegularNX":
            reservoir_initializer = RegularNX(
                degree= reservoir_degree,
                spectral_radius= spectral_radius,
                sigma= reservoir_sigma,
            )
        case "ErdosRenyi":
            reservoir_initializer = ErdosRenyi(
                degree= reservoir_degree,
                spectral_radius= spectral_radius,
                sigma= reservoir_sigma,
            )
        case "WattsStrogatzOwn":
            reservoir_initializer = WattsStrogatzOwn(
                degree= reservoir_degree,
                spectral_radius= spectral_radius,
                rewiring_p= rewiring,
                sigma= reservoir_sigma,
            )
        case "WattsStrogatzNX":
            reservoir_initializer = WattsStrogatzNX(
                degree= reservoir_degree,
                spectral_radius= spectral_radius,
                rewiring_p= rewiring,
                sigma= reservoir_sigma,
            )

    ############### CHOOSE THE MODEL ###############

    match model:
        case "ESN":
            model = ESN(
                units= units,
                leak_rate= leak_rate,
                input_reservoir_init=input_initializer,
                input_bias_init=input_bias_initializer,
                reservoir_kernel_init=reservoir_initializer,
                esn_activation=reservoir_activation,
            )

        case "Parallel-ESN":
            model = ParallelESN(
                units_per_reservoir= units,
                reservoir_amount=reservoir_amount,
                overlap=overlap,
                leak_rate= leak_rate,
                input_reservoir_init=input_initializer,
                input_bias_init=input_bias_initializer,
                reservoir_kernel_init=reservoir_initializer,
                esn_activation=reservoir_activation,
            )

        case "Reservoir":
            print("Yet to be implemented")
            return

    ############### CHOOSE THE READOUT LAYER ###############

    match readout_layer:
        case "linear":
            model = linear_readout(
                model=model,
                transient_data=transient_data,
                train_data=train_data,
                train_target=train_target,
                regularization= regularization,
            )

        case "sgd":
            print("Yet to be implemented")
            return
        case "mlp":
            print("Yet to be implemented")
            return

    ############### SAVING TRAINED MODEL ###############

    if save_model:

        if output_dir is None:
            os.makedirs("Models", exist_ok=True)

        if file_name is not None:
            model_name = join(output_dir, file_name)
        else:
            model_name = join(output_dir, f"/{model.model.seed}")

        # Save the model and save the parameters dictionary in a json file inside the model folder
   
        model.save(model_name)

        with open(
            join(model_name, "params.json"),
            "w",
            encoding="utf-8",
        ) as _f_:
            json.dump(params, _f_)
    
    return model, params



def _forecast(
    
    trained_model,
    data_file: str,
    output_dir: str,
    model_params: dict = {},

    # Forecast params
    forecast_method: str = 'classic',
    forecast_length: int = 1000,
    section_initialization_length: int = 50,
    number_of_sections: int = 10,

    # Charge saved model
    load_saved_model = False

):
    """Load a model and forecast the data.

    Args:
        trained_model (str): The trained model to be used for forecasting
        data_file (str): The data file to be used for training the model
        output_dir (str):
        model_params (dict) = {},

        forecast_method (str): The method to be used for forecasting. The default is ClassicForecast.
        forecast_length (int): The number of points to be forecasted. The default is 1000.
        section_initialization_length: int = 50,
        number_of_sections: int = 10,

        load_saved_model (bool): True -> the model will be load of a file so trained_model will be a path
                               | False -> Not need to load the model, trained_model will be a ESN_model

    Returns:
        None

    """

    transient
    train_length
    init_transient
    
    if load_saved_model:

        # Load the param json from the model location
        params = load_model_json(trained_model)
        model = load_model(trained_model, compile=False)

        transient= params["transient"],
        train_length= params["train_length"],
        init_transient= params["init_transient"]

    else:

        model = trained_model

        transient= model_params["transient"],
        train_length= model_params["train_length"],
        init_transient= model_params["init_transient"]

        
    # Load the data
    (
        _,
        _,
        _,
        forecast_transient_data,
        val_data,
        val_target,
    ) = load_data (
        data_file,
        transient= transient,
        train_length= train_length,
        init_transient= init_transient
    )

    # Load the model

    ############### CHOOSE THE FORECAST METHOD AND FORECAST ###############

    match forecast_method:
        case "classic":
            predictions = classic_forecast(
                model,
                forecast_transient_data,
                val_data,
                val_target,
                forecast_length=forecast_length,
            )
            # this will be of shape (1, forecast_length, features) I need to reshape it to (forecast_length, features)
            predictions = predictions[0]

        case "section":
            predictions = section_forecast(
                model,
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
    trained_model_name = trained_model.split("/")[-1]

    data_name = data_file.split("/")[-1]
    print(data_name)

    if not os.path.exists(f"forecasts/{trained_model_name}"):
        os.makedirs(f"forecasts/{trained_model_name}")

    # Save the forecasted data as csv using pandas
    pd.DataFrame(predictions).to_csv(
        f"{output_dir}/{trained_model_name}/{data_name}_{forecast_method}_forecasted.csv",
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
    init_transient,
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
        init_transient=init_transient,
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
