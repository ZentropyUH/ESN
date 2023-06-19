import json
import os
from os.path import join

# To avoid tensorflow verbosity
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from pathlib import Path

import numpy as np
import pandas as pd

# This is because they still conserve the old API for tf 1.x
try:
    from keras.initializers.initializers import Zeros
except ModuleNotFoundError:
    from keras.initializers import Zeros

from keras.models import load_model

from src.customs.custom_initializers import (
    ErdosRenyi,
    InputMatrix,
    RandomUniform,
    RegularNX,
    RegularOwn,
    WattsStrogatzNX,
    WattsStrogatzOwn,
)
from src.customs.custom_models import ESN, ParallelESN, ReservoirModel
from src.forecasters import classic_forecast, section_forecast
from src.plotters import (
    plot_contourf_forecast,
    plot_linear_forecast,
    plot_rmse,
    render_video,
)
from src.readout_generators import linear_readout
from src.utils import get_name_from_dict, get_range, load_data


def training(
    # General params
    model,
    units,
    input_initializer,
    input_bias_initializer,
    input_scaling,
    leak_rate,
    reservoir_activation,
    # Classic Cases
    spectral_radius,
    reservoir_initializer,
    rewiring,
    reservoir_degree,
    reservoir_sigma,
    # Parallel cases
    reservoir_amount,
    overlap,
    # Readout params
    readout_layer,
    regularization,
    # Training params
    init_transient,
    transient,
    train_length,
    data_file,
    output_dir,
    trained_name,
):
    """
    Trains an Echo State Network on the data provided in the data file.

    The data file should be a csv file with the rows being the time and the columns being the dimensions. The data file should be provided with full path.
    """
    ################ GET THE PARAMETERS WITH POSSIBLE RANGES ################

    params = locals().copy()

    # General params
    units = get_range(units, step=1000, method="linear")
    units = [int(unit) for unit in units]

    input_scaling = get_range(input_scaling)
    leak_rate = get_range(leak_rate)
    spectral_radius = get_range(spectral_radius)

    reservoir_degree = get_range(reservoir_degree)
    reservoir_degree = [int(degree) for degree in reservoir_degree]

    reservoir_sigma = get_range(reservoir_sigma)
    rewiring = get_range(rewiring)
    # This will typically be chosen to be 1e-4
    regularization = get_range(regularization, method="log", base=10)

    train_length = get_range(train_length)
    train_length = [int(length) for length in train_length]

    ## INPUT INITIALIZER

    for _units in units:
        for _input_scaling in input_scaling:
            for _leak_rate in leak_rate:
                for _spectral_radius in spectral_radius:
                    for _reservoir_degree in reservoir_degree:
                        for _reservoir_sigma in reservoir_sigma:
                            for _rewiring in rewiring:
                                for _regularization in regularization:
                                    for _train_length in train_length:
                                        ############### LOAD THE DATA ###############

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
                                            transient=transient,
                                            train_length=_train_length,
                                            init_transient=init_transient,
                                        )

                                        ############### CHOOSE THE INPUT INITIALIZER ###############

                                        match input_initializer:
                                            case "InputMatrix":
                                                input_initializer = (
                                                    InputMatrix(
                                                        sigma=_input_scaling
                                                    )
                                                )
                                            case "RandomUniform":
                                                input_initializer = (
                                                    RandomUniform(
                                                        sigma=_input_scaling
                                                    )
                                                )

                                        ############### CHOOSE THE INPUT INITIALIZER ###############

                                        match input_bias_initializer:
                                            case "InputMatrix":
                                                input_bias_initializer = (
                                                    InputMatrix(
                                                        sigma=_input_scaling
                                                    )
                                                )
                                            case "RandomUniform":
                                                input_bias_initializer = (
                                                    RandomUniform(
                                                        sigma=_input_scaling
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
                                                    degree=_reservoir_degree,
                                                    spectral_radius=_spectral_radius,
                                                    sigma=_reservoir_sigma,
                                                )
                                            case "RegularNX":
                                                reservoir_initializer = RegularNX(
                                                    degree=_reservoir_degree,
                                                    spectral_radius=_spectral_radius,
                                                    sigma=_reservoir_sigma,
                                                )
                                            case "ErdosRenyi":
                                                reservoir_initializer = ErdosRenyi(
                                                    degree=_reservoir_degree,
                                                    spectral_radius=_spectral_radius,
                                                    sigma=_reservoir_sigma,
                                                )
                                            case "WattsStrogatzOwn":
                                                reservoir_initializer = WattsStrogatzOwn(
                                                    degree=_reservoir_degree,
                                                    spectral_radius=_spectral_radius,
                                                    rewiring_p=_rewiring,
                                                    sigma=_reservoir_sigma,
                                                )
                                            case "WattsStrogatzNX":
                                                reservoir_initializer = WattsStrogatzNX(
                                                    degree=_reservoir_degree,
                                                    spectral_radius=_spectral_radius,
                                                    rewiring_p=_rewiring,
                                                    sigma=_reservoir_sigma,
                                                )

                                        ############### CHOOSE THE MODEL ###############

                                        match model:
                                            case "ESN":
                                                model = ESN(
                                                    units=_units,
                                                    leak_rate=_leak_rate,
                                                    input_reservoir_init=input_initializer,
                                                    input_bias_init=input_bias_initializer,
                                                    reservoir_kernel_init=reservoir_initializer,
                                                    esn_activation=reservoir_activation,
                                                )

                                            case "Parallel-ESN":
                                                model = ParallelESN(
                                                    units_per_reservoir=_units,
                                                    reservoir_amount=reservoir_amount,
                                                    overlap=overlap,
                                                    leak_rate=_leak_rate,
                                                    input_reservoir_init=input_initializer,
                                                    input_bias_init=input_bias_initializer,
                                                    reservoir_kernel_init=reservoir_initializer,
                                                    esn_activation=reservoir_activation,
                                                )

                                            case "Reservoir_to_be_implemented":
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
                                                    regularization=_regularization,
                                                )

                                            case "sgd":
                                                print("Yet to be implemented")
                                                return
                                            case "mlp":
                                                print("Yet to be implemented")
                                                return

                                        ############### SAVING TRAINED MODEL ###############

                                        # Prune path from data_file
                                        data_file_name = data_file.split("/")[
                                            -1
                                        ]

                                        # Choose only the most important parameters to name the model
                                        name_dict = {
                                            "0mdl": locals()["model"],
                                            "units": _units,
                                            "sigma": _input_scaling,
                                            "sr": _spectral_radius,
                                            "degr": _reservoir_degree,
                                            "resigma": _reservoir_sigma,
                                            "rw": _rewiring,
                                            "reg": _regularization,
                                            "readl": readout_layer,
                                            "dta": data_file_name,
                                        }

                                        model_path = join(
                                            output_dir,
                                            get_name_from_dict(name_dict),
                                        )

                                        if trained_name is not None:
                                            model_path = join(
                                                output_dir, trained_name
                                            )

                                        # Save the model and save the parameters dictionary in a json file inside the model folder
                                        model.save(model_path)

                                        with open(
                                            join(
                                                output_dir,
                                                "params.json",
                                                encoding="utf-8",
                                            ),
                                            "w",
                                        ) as f:
                                            json.dump(
                                                params,
                                                f,
                                                indent=4,
                                                sort_keys=True,
                                                separators=(",", ": "),
                                            )


def forecasting(
    forecast_method: str,
    forecast_length: int,
    section_initialization_length: int,
    number_of_sections: int,
    init_transient: int,
    transient: int,
    train_length: int,
    output_dir: str,
    trained_model: str,
    data_file: str,
    forecast_name: str,
):
    """
    Load a model and forecast the data.

    Args:
        forecast_method (str): The method to be used for forecasting. The default is ClassicForecast.
        forecast_length (int): The number of points to be forecasted. The default is 1000.
        trained_model (str): The trained model to be used for forecasting
        data_file (str): The data file to be used for training the model

    Returns:
        None

    """

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
        init_transient=init_transient,
    )

    # Load the model
    model = load_model(trained_model, compile=False)

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
    # Prune path from data_file
    data_file_name = data_file.split("/")[-1]

    # Prune path from trained_model
    if forecast_name is None:
        trained_model_name = (
            trained_model.split("/")[-1] + f"_{forecast_method}_forecasted"
        )
    else:
        trained_model_name = forecast_name

    # Save the forecasted data as csv using pandas
    pd.DataFrame(predictions).to_csv(
        join(output_dir, trained_model_name + ".csv"),
        index=False,
        header=None,
    )


def plot(
    plot_type,
    predictions,
    data_file,
    lyapunov_exponent,
    delta_time,
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
