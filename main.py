#!/usr/bin/python3
import json
import os

from pathlib import Path

import click
import numpy as np
import pandas as pd

# This is because they still conserve the old API for tf 1.x
from keras.initializers.initializers import Zeros
from keras.models import load_model
from tqdm import tqdm

from custom_initializers import (
    ErdosRenyi,
    InputMatrix,
    RandomUniform,
    RegularNX,
    RegularOwn,
    WattsStrogatzNX,
    WattsStrogatzOwn,
)
from custom_models import ESN, ParallelESN, ReservoirModel
from forecasters import classic_forecast, section_forecast
from plotters import (
    plot_contourf_forecast,
    plot_linear_forecast,
    plot_rmse,
    render_video,
)
from readout_generators import linear_readout, sgd_linear_readout
from utils import get_name_from_dict, get_range, load_data, load_model_json

# To avoid tensorflow verbosity
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """A command line for training and making predictions with general ESN-like models from provided dynamical systems timeseries."""


# Train command
@cli.command()

################ GENERAL RESERVOIR PARAMETERS ################


@click.option(
    "--model",
    "-m",
    type=click.Choice(["ESN", "Parallel-ESN", "Reservoir_to_be_implemented"]),
    default="ESN",
    help="The model to be used. The default is ESN.",
)
@click.option(
    "--units",
    "-u",
    type=click.STRING,
    default="2000",
    help="The number of units in the reservoir. The default is 2000. If a range of values is given, the script will be executed the specified number of times with different values of the number of units. The values will be chosen linearly between the first and the second value. If a list of values is given, the script will be executed the specified number of times with the values in the list.",
)
@click.option(
    "--input-initializer",
    "-ii",
    type=click.Choice(["InputMatrix", "RandomUniform"]),
    default="InputMatrix",
    help="The initializer for the input weights. The default is InputMatrix.",
)
@click.option(
    "--input-scaling",
    "-is",
    type=click.STRING,
    default="0.5",
    help="The input scaling parameter. The default is 0.5. If a range of values is given, the script will be executed the specified number of times with different values of the input scaling parameter. The values will be chosen linearly between the first and the second value. If a list of values is given, the script will be executed the specified number of times with the values in the list.",
)
@click.option(
    "--input-bias-initializer",
    "-ib",
    type=click.Choice(["InputMatrix", "RandomUniform", "None"]),
    default="RandomUniform",
    help="The initializer for the input bias weights. The default is RandomUniform.",
)
@click.option(
    "--leak-rate",
    "-lr",
    type=click.STRING,
    default="1.",
    help="The leak rate of the reservoir. The default is 1. If a range of values is given, the script will be executed the specified number of times with different values of the leak rate. The values will be chosen linearly between the first and the second value. If a list of values is given, the script will be executed the specified number of times with the values in the list.",
)
@click.option(
    "--reservoir-activation",
    "-a",
    type=click.Choice(["tanh", "relu", "sigmoid", "identity"]),
    default="tanh",
    help="The activation function of the reservoir. The default is tanh. Only used if ESN or Parallel_ESN is used.",
    callback=lambda ctx, param, value: value
    if ctx.params["model"] == "ESN" or ctx.params["model"] == "Parallel-ESN"
    else None,
)

################ CLASSIC ESN MODELS ONLY PARAMETERS


@click.option(
    "--spectral-radius",
    "-sr",
    type=click.STRING,
    default="0.99",
    help="The spectral radius of the reservoir. The default is 0.99. Only used if ESN or Parallel_ESN is used. If a range of values is given, the script will be executed the specified number of times with different values of the spectral radius. The values will be chosen linearly between the first and the second value. If a list of values is given, the script will be executed the specified number of times with the values in the list.",
    # This decides to use this parameter if ESN or Parallel-ESN is chosen as model, else is discarded
    callback=lambda ctx, param, value: value
    if ctx.params["model"] == "ESN" or ctx.params["model"] == "Parallel-ESN"
    else None,
)

# Check when are we needing this
@click.option(
    "--reservoir-initializer",
    "-ri",
    type=click.Choice(
        [
            "RegularOwn",
            "RegularNX",
            "ErdosRenyi",
            "WattsStrogatzOwn",
            "WattsStrogatzNX",
        ]
    ),
    default="WattsStrogatzOwn",
    help="The initializer for the reservoir weights. The default is WattsStrogatzOwn. Only used if ESN or Parallel_ESN is used.",  # Maybe play later with topologies on ECA and Oscillators. First we have to study impact on EOC.allow_from_autoenv=
    callback=lambda ctx, param, value: value
    if ctx.params["model"] == "ESN" or ctx.params["model"] == "Parallel-ESN"
    else None,
)
@click.option(
    "--rewiring",
    "-rw",
    type=click.STRING,
    default="0.5",
    help="The rewiring probability of the WattsStrogatz graph. The default is 0.5. Only used if ESN or Parallel_ESN is used. If a range of values is given, the script will be executed the specified number of times with different values of the degree parameter. The values will be chosen linearly between the first and the second value. If a list of values is given, the script will be executed the specified number of times with the values in the list.",
    callback=lambda ctx, param, value: value
    if ctx.params["reservoir_initializer"] == "WattsStrogatzOwn"
    or ctx.params["reservoir_initializer"] == "WattsStrogatzNX"
    else None,
)
@click.option(
    "--reservoir-degree",
    "-rd",
    type=click.STRING,
    default="3",
    help="The degree of the reservoir. The default is 3. Only used if ESN or Parallel_ESN is used. If a range of values is given, the script will be executed the specified number of times with different values of the degree parameter. The values will be chosen linearly between the first and the second value. If a list of values is given, the script will be executed the specified number of times with the values in the list.",
    callback=lambda ctx, param, value: value
    if ctx.params["model"] == "ESN" or ctx.params["model"] == "Parallel-ESN"
    else None,
)
@click.option(
    "--reservoir-sigma",
    "-rs",
    type=click.STRING,
    default="0.5",
    help="The standard deviation for the reservoir weights. The default is 0.5. Only used if ESN or Parallel_ESN is used. If a range of values is given, the script will be executed the specified number of times with different values of the sigma parameter. The values will be chosen linearly between the first and the second value. If a list of values is given, the script will be executed the specified number of times with the values in the list.",
    callback=lambda ctx, param, value: value
    if ctx.params["model"] == "ESN" or ctx.params["model"] == "Parallel-ESN"
    else None,
)

################ PARALLEL SCHEME PARAMETERS ################


@click.option(
    "--reservoir-amount",
    "-ra",
    type=click.INT,
    default=10,
    help="The number of reservoirs to be used. The default is 10. Only used if Parallel_ESN is used or other parallel scheme.",
    # This decides to use this parameter if Parallel-ESN is chosen as model, else is discarded
    callback=lambda ctx, param, value: value
    if ctx.params["model"] == "Parallel-ESN"
    else None,
)
@click.option(
    "--overlap",
    "-ol",
    type=click.STRING,
    default="6",
    help="The number of overlapping units between reservoirs. The default is 6. Only used if Parallel_ESN is used or other parallel scheme. If a range of values is given, the script will be executed the specified number of times with different values of the overlap parameter. The values will be chosen linearly between the first and the second value. If a list of values is given, the script will be executed the specified number of times with the values in the list.",
    # This decides to use this parameter if Parallel-ESN is chosen as model, else is discarded
    callback=lambda ctx, param, value: value
    if ctx.params["model"] == "Parallel-ESN"
    else None,
)

##################################################################################################################


################ READOUT PARAMETERS ################


@click.option(
    "--readout-layer",
    "-rl",
    type=click.Choice(["linear", "sgd", "mlp"]),
    default="linear",
    help="The type of readout layer of the model; 'linear' if the layer is a linear regression using Ridge (Tikhonov) regularization scheme; 'sgd' if the readout should be a linear regression to be calculated iteratively with stochastic gradient descent; 'mlp' if the readout is to be chosen as a multilayer perceptron. If 'mlp' is chosen more options should be provided.",
)
@click.option(
    "--regularization",
    "-rg",
    type=click.STRING,
    default="1e-4",
    help="The regularization parameter. The default is 1e-4. If a range of values is given, the script will be executed the specified number of times with different values of the regularization parameter. The values will be chosen logarithmically between the first and the second value. If a list of values is given, the script will be executed the specified number of times with the values in the list.",
)

##################################################################################################################

################ TRAINING PARAMETERS ################


@click.option(
    "--init-transient",
    "-it",
    type=click.INT,
    default=1000,
    help="The number of transient points to be discarded. The default is 1000.",
)
@click.option(
    "--transient",
    "-tr",
    type=click.INT,
    default=1000,
    help="The number of transient points to be discarded. The default is 1000.",
)
@click.option(
    "--train-length",
    "-tl",
    type=click.STRING,
    default="10000",
    help="The number of points to be used for training. The default is 10000. If a range of values is given, the script will be executed the specified number of times worg.freedesktop.PackageKit.proxyith different values of the training length. The values will be chosen linearly between the first and the second value. If a list of values is given, the script will be executed the specified number of times with the values in the list.",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(exists=True),
    default="Models",
    help="The directory where the results will be saved. The default is the current directory.",
)
@click.option(
    "--data-file",
    "-df",
    type=click.Path(exists=True),
    help="Data file to be used for training.",
)
@click.pass_context
##################################################################################################################
def train(
    ctx,
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
):
    """
    Trains an Echo State Network on the data provided in the data file.

    The data file should be a csv file with the rows being the time and the columns being the dimensions. The data file should be provided with full path.
    """
    ################ GET THE PARAMETERS WITH POSSIBLE RANGES ################

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

    for _units in tqdm(units, postfix="Units"):
        for _input_scaling in tqdm(input_scaling, postfix="Input Scaling"):
            for _leak_rate in tqdm(leak_rate, postfix="Leak rate"):
                for _spectral_radius in tqdm(
                    spectral_radius, postfix="Spectral radius"
                ):
                    for _reservoir_degree in tqdm(
                        reservoir_degree, postfix="Degree"
                    ):
                        for _reservoir_sigma in tqdm(
                            reservoir_sigma, postfix="Reservoir std"
                        ):
                            for _rewiring in tqdm(
                                rewiring, postfix="Rewiring"
                            ):
                                for _regularization in tqdm(
                                    regularization, postfix="regularization"
                                ):
                                    for _train_length in tqdm(
                                        train_length, postfix="Train length"
                                    ):
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

                                        params = ctx.__dict__["params"]

                                        # Prune path from data_file
                                        data_file_name = data_file.split("/")[
                                            -1
                                        ]

                                        # Choose only the most important parameters to name the model
                                        name_dict = {
                                            "0mdl": ctx.__dict__["params"][
                                                "model"
                                            ],
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

                                        model_name = (
                                            output_dir
                                            + f"/{get_name_from_dict(name_dict)}"
                                        )
                                        # Save the model and save the parameters dictionary in a json file inside the model folder
                                        model.save(
                                            model_name
                                        )
                                        with open(
                                            model_name + "/params.json",
                                            "w",
                                            encoding="utf-8",
                                        ) as _f_:
                                            json.dump(params, _f_)


################ FORECAST PARAMETERS ################


@cli.command()
@click.option(
    "--forecast-method",
    "-fm",
    type=click.Choice(["classic", "section"]),
    default="classic",
    help="The method to be used for forecasting. The default is Classic.",
)
@click.option(
    "--forecast-length",
    "-fl",
    type=click.INT,
    default=1000,
    help="The number of points to be forecasted. The default is 1000.",
)

################ ONLY SECTION-FORECAS PARAMETERS ################


@click.option(
    "--section-initialization-length",
    "-sil",
    type=click.INT,
    default=50,
    help="The number of points to be used for initializing the sections with true data. The default is 50.",
    callback=lambda ctx, param, value: value
    if ctx.params["forecast_method"] == "classic"
    or ctx.params["forecast_method"] == "section"
    else None,
)
@click.option(
    "--number-of-sections",
    "-nos",
    type=click.INT,
    default=10,
    help="The number of sections to be used for forecasting. The default is 10.",
    callback=lambda ctx, param, value: value
    if ctx.params["forecast_method"] == "classic"
    or ctx.params["forecast_method"] == "section"
    else None,
)


#################################################################


@click.option(
    "--trained-model",
    "-tm",
    type=click.Path(exists=True),
    help="The trained model to be used for forecasting",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(exists=True),
    default=".",
    help="The output directory where the forecasted data will be saved",
)
@click.option(
    "--data-file",
    "-df",
    type=click.Path(exists=True),
    help="The data file to be used for training the model",
)
def forecast(
    forecast_method: str,
    forecast_length: int,
    section_initialization_length: int,
    number_of_sections: int,
    output_dir: str,
    trained_model: str,
    data_file: str,
):
    """Load a model and forecast the data.

    Args:
        forecast_method (str): The method to be used for forecasting. The default is ClassicForecast.
        forecast_length (int): The number of points to be forecasted. The default is 1000.
        trained_model (str): The trained model to be used for forecasting
        data_file (str): The data file to be used for training the model

    Returns:
        None

    """

    # Load the param json from the model location
    params = load_model_json(trained_model)

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
        transient=int(params["transient"]),
        train_length=int(params["train_length"]),
        init_transient=int(params["init_transient"]),
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
    trained_model_name = trained_model.split("/")[-1]

    # Save the forecasted data as csv using pandas
    pd.DataFrame(predictions).to_csv(
        f"{output_dir}/{trained_model_name}_{forecast_method}_forecasted.csv",
        index=False,
        header=None,
    )


# plot command that receives a plot_type, prediction file and a data file and makes the plot
@cli.command()
@click.option(
    "--lyapunov-exponent",
    "-le",
    type=click.FLOAT,
    default=1,
    help="The lyapunov exponent of the data. The default is 1. It is used to scale the time to the lyapunov time units.",
)
@click.option(
    "--plot-type",
    "-pt",
    type=click.Choice(["linear", "contourf", "rmse", "video"]),
    default="linear",
    help="The type of plot to be made. The default is linear.",
)
@click.option(
    "--delta-time",
    "-dt",
    type=click.FLOAT,
    default=1,
    help="The time step between each point in the data. The default is 1.",
)
@click.option(
    "--title",
    "-T",
    type=click.STRING,
    default="",
    help="The title of the plot. The default is empty.",
)
@click.option(
    "--save-path",
    "-sp",
    type=click.Path(exists=True),
    default=".",
    help="The path where the plot will be saved. The default is None.",
)
@click.option(
    "--show/--no-show",
    "-s",
    type=click.BOOL,
    default=True,
    help="Whether to show the plot or not. The default is True.",
)
@click.option(
    "--y-labels",
    "-yl",
    type=click.STRING,
    default=None,
    help="The labels of the y axis. The default is None. Value should be a string with the labels separated by commas.",
)
@click.option(
    "--x-label",
    "-xl",
    type=click.STRING,
    default="t",
    help="The label of the x axis. The default is time (t).",
)
@click.option(
    "--y-values",
    "-yv",
    type=click.STRING,
    default=None,
    help="The values of the y axis. The default is None. Value should be a string with both values separated by commas.",
)


# This is to know the validation data to be able to compare the predictions with the actual data and not with the training data.
@click.option(
    "--init-transient",
    "-it",
    type=click.INT,
    help="The number of transient points that were discarded at the beginning of the data.",
)
@click.option(
    "--transient",
    "-tr",
    type=click.INT,
    help="The number of transient points discarded in the training of the model.",
)
@click.option(
    "--train-length",
    "-tl",
    type=click.INT,
    help="The number of points used for the training of the model.",
)
@click.option(
    "predictions",
    "-pr",
    type=click.Path(exists=True),
    help="The path to the file containing the predictions.",
)
@click.option(
    "--data_file",
    "-df",
    type=click.Path(exists=True),
    help="The path to the file containing the data.",
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


if __name__ == "__main__":
    cli()
