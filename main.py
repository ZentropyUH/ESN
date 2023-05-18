#!/usr/bin/python3
"""CLI interface to train and forecast with ESN models."""

# pylint: disable=unused-argument
# pylint: disable=line-too-long
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import click

from src.functions import training, forecasting


class Config:
    def __init__(self):
        self.verbose = False


from model_functions import _forecast, _plot, _train

# To avoid tensorflow verbosity
pass_config = click.make_pass_decorator(Config, ensure=True)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose mode.")
@click.version_option(version="1.0.0")
@click.pass_context
def cli(ctx, verbose):
    """
    A command line for training and making predictions with general ESN-like models from provided dynamical systems timeseries.
    """
    ctx.obj = Config()
    ctx.obj.verbose = verbose

# region Train params
@cli.command()


################ GENERAL RESERVOIR PARAMETERS ################
@click.option(
    "--model",
    "-m",
    type=click.Choice(["ESN", "Parallel-ESN", "Reservoir"]),
    default="ESN",
    help="The model to be used. The default is ESN.",
)
@click.option(
    "--units",
    "-u",
    type=click.STRING,
    default="2000",
    help="The number of units in the reservoir."\
        " The default is 2000. If a range of values is given, the script will be executed the "\
        "specified number of times with different values of the number of units. "\
        "The values will be chosen linearly between the first and the second value. "\
        "If a list of values is given, the script will be executed the specified number of "\
        "times with the values in the list.",
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
# endregion

# region classic model params
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
# endregion

# region parallel model params
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
# endregion

# region readout params
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
# endregion

# region training params
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
@click.option(
    "--trained-name",
    "-tn",
    type=click.STRING,
    default=None,
    help="Training folder name.",
)
# endregion
def train(
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
    """Train a specific model on a given data file."""
    _train(**locals())


# region Forecast params
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
# endregion

# region single section params
################ SINGLE SECTION-FORECAS PARAMETERS ################
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

################ PARAMETERS TO EXTRACT THE VALIDATION DATA/TARGET ################


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

# endregion

# region General forecast params
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
@click.option(
    "--forecast-name",
    "-fn",
    type=click.STRING,
    default=None,
    help="Forecast file name.",
)
# endregion
def forecast(
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
    """Load a model and forecast the data.

    Args:
        forecast_method (str): The method to be used for forecasting. The default is ClassicForecast.
        forecast_length (int): The number of points to be forecasted. The default is 1000.
        trained_model (str): The trained model to be used for forecasting
        data_file (str): The data file to be used for training the model

    Returns:
        None

    """

    # Prune path from trained_model
    if forecast_name is None:
        trained_model_name = trained_model.split("/")[-1] + f"_{forecast_method}_forecasted"
    else:
        trained_model_name = forecast_name

    # Prune path from trained_model
    if forecast_name is None:
        trained_model_name = trained_model.split("/")[-1] + f"_{forecast_method}_forecasted"
    else:
        trained_model_name = forecast_name

    """Make predictions with a given model on a data file."""
    _forecast(**locals())


# region Plot params

# region general params
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
    "--plot-points",
    "-pp",
    type=click.INT,
    default=None,
    help="The number of points/steps to plot."
)
@click.option(
    "--y-values",
    "-yv",
    type=click.STRING,
    default=None,
    help="The values of the y axis. The default is None. Value should be a string with both values separated by commas.",
)
# endregion

# region datafiles params

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
# endregion
# region training-related params

# This is to know the validation data to be able to compare the predictions with the actual data and not with the training data.
@click.option(
    "--init-transient",
    "-it",
    type=click.INT,
    help="The number of transient points that were discarded at the beginning of the data.",
    default=1000,
)
@click.option(
    "--transient",
    "-tr",
    type=click.INT,
    help="The number of transient points discarded in the training of the model.",
    default=1000,
)
@click.option(
    "--train-length",
    "-tl",
    type=click.INT,
    help="The number of points used for the training of the model.",
    default=10000,
)

# endregion

# endregion
def plot(
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
    """Plot different data."""
    _plot(**locals())


if __name__ == "__main__":
    cli()

