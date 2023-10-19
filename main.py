#!/usr/bin/python3
import os
# To eliminate tensorflow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import typer
from typing import List

from t_utils import *
from functions import _train
from functions import _forecast_from_saved_model
from functions import _plot
from slurm_grid.grid import _slurm_grid
from slurm_grid.tools import _best_results
from slurm_grid.tools import _results_data
from slurm_grid.tools import _search_unfinished_combinations
from slurm_grid.tools import _init_slurm_grid
from slurm_grid.tools import _grid_aux

app = typer.Typer()


@app.command()
def train(
    # region General params
    # General params
    data_file: str = typer.Option(
        ...,
        "--data-file",
        "-df",
        help="Data file to be used for training.",
    ),
    output_dir: str = typer.Option(
        ...,
        "--output-dir",
        "-o",
        help="The directory where the results will be saved. The default is the current directory.",
    ),
    model: EModel = typer.Option("ESN", "--model", "-m", help=""),
    units: int = typer.Option(..., "--units", "-u", help=""),
    input_initializer: EInputInitializer = typer.Option(
        "InputMatrix",
        "--input-initializer",
        "-ii",
        help="",
    ),
    input_bias_initializer: InputBiasInitializer = typer.Option(
        "RandomUniform",
        "--input-bias-initializer",
        "-ib",
        help="The initializer for the input bias weights. The default is RandomUniform.",
    ),
    input_scaling: float = typer.Option(
        0.5,
        "--input-scaling",
        "-is",
        help="The input scaling parameter. The default is 0.5. If a range of values is given, the script will be executed the specified number of times with different values of the input scaling parameter. The values will be chosen linearly between the first and the second value. If a list of values is given, the script will be executed the specified number of times with the values in the list.",
    ),
    leak_rate: float = typer.Option(
        1.0,
        "--leak-rate",
        "-lr",
        help="The leak rate of the reservoir. The default is 1. If a range of values is given, the script will be executed the specified number of times with different values of the leak rate. The values will be chosen linearly between the first and the second value. If a list of values is given, the script will be executed the specified number of times with the values in the list.",
    ),
    reservoir_activation: ReservoirActivation = typer.Option(
        "tanh",
        "--reservoir-activation",
        "-a",
        help="The activation function of the reservoir. The default is tanh. Only used if ESN or Parallel_ESN is used.",
    ),
    seed: int = typer.Option(
        None,
        "--seed",
        "-s",
        help="The seed to be used for the random number generator. If not specified, it is randomly generated.",
    ),
    # endregion
    # Classic Cases
    spectral_radius: float = typer.Option(
        0.99,
        "--spectral-radius",
        "-sr",
        help="The spectral radius of the reservoir. The default is 0.99. Only used if ESN or Parallel_ESN is used. If a range of values is given, the script will be executed the specified number of times with different values of the spectral radius. The values will be chosen linearly between the first and the second value. If a list of values is given, the script will be executed the specified number of times with the values in the list.",
    ),
    reservoir_initializer: ReservoirInitializer = typer.Option(
        "WattsStrogatzNX",
        "--reservoir-initializer",
        "-ri",
        help="The initializer for the reservoir weights. The default is WattsStrogatzNX. Only used if ESN or Parallel_ESN is used.",  # Maybe play later with topologies on ECA and Oscillators. First we have to study impact on EOC.allow_from_autoenv=
    ),
    rewiring: float = typer.Option(
        0.5,
        "--rewiring",
        "-rw",
        help="The rewiring probability of the WattsStrogatz graph. The default is 0.5. Only used if ESN or Parallel_ESN is used. If a range of values is given, the script will be executed the specified number of times with different values of the degree parameter. The values will be chosen linearly between the first and the second value. If a list of values is given, the script will be executed the specified number of times with the values in the list.",
    ),
    reservoir_degree: int = typer.Option(
        3,
        "--reservoir-degree",
        "-rd",
        help="The degree of the reservoir. The default is 3. Only used if ESN or Parallel_ESN is used. If a range of values is given, the script will be executed the specified number of times with different values of the degree parameter. The values will be chosen linearly between the first and the second value. If a list of values is given, the script will be executed the specified number of times with the values in the list.",
    ),
    reservoir_sigma: float = typer.Option(
        0.5,
        "--reservoir-sigma",
        "-rs",
        help="The standard deviation for the reservoir weights. The default is 0.5. Only used if ESN or Parallel_ESN is used. If a range of values is given, the script will be executed the specified number of times with different values of the sigma parameter. The values will be chosen linearly between the first and the second value. If a list of values is given, the script will be executed the specified number of times with the values in the list.",
    ),
    # Parallel cases
    reservoir_amount: int = typer.Option(
        10,
        "--reservoir-amount",
        "-ra",
        help="The number of reservoirs to be used. The default is 10. Only used if Parallel_ESN is used or other parallel scheme.",
    ),
    overlap: int = typer.Option(
        6,
        "--overlap",
        "-ol",
        help="The number of overlapping units between reservoirs. The default is 6. Only used if Parallel_ESN is used or other parallel scheme. If a range of values is given, the script will be executed the specified number of times with different values of the overlap parameter. The values will be chosen linearly between the first and the second value. If a list of values is given, the script will be executed the specified number of times with the values in the list.",
    ),
    # Readout params
    readout_layer: ReadoutLayer = typer.Option(
        "linear",
        "--readout-layer",
        "-rl",
        help="The type of readout layer of the model; 'linear' if the layer is a linear regression using Ridge (Tikhonov) regularization scheme; 'sgd' if the readout should be a linear regression to be calculated iteratively with stochastic gradient descent; 'mlp' if the readout is to be chosen as a multilayer perceptron. If 'mlp' is chosen more options should be provided.",
    ),
    regularization: float = typer.Option(
        1e-4,
        "--regularization",
        "-rg",
        help="The regularization parameter. The default is 1e-4. If a range of values is given, the script will be executed the specified number of times with different values of the regularization parameter. The values will be chosen logarithmically between the first and the second value. If a list of values is given, the script will be executed the specified number of times with the values in the list.",
    ),
    # Training params
    transient: int = typer.Option(
        1000,
        "--transient",
        "-tr",
        help="The number of transient points to be discarded. The default is 1000.",
    ),
    train_length: int = typer.Option(
        10000,
        "--train-length",
        "-tl",
        help="The number of points to be used for training. The default is 10000. If a range of values is given, the script will be executed the specified number of times worg.freedesktop.PackageKit.proxyith different values of the training length. The values will be chosen linearly between the first and the second value. If a list of values is given, the script will be executed the specified number of times with the values in the list.",
    ),
    steps: int = typer.Option(
        1,
        "--steps",
        "-s",
        help="Number of steps among data point to ignore. Used to variate the data dt.",
    ),
):
    """Train a specific model on a given data file."""
    _train(**locals())


@app.command()
def forecast(
    trained_model_path: str = typer.Option(
        ...,
        "--trained-model-path",
        "-tm",
        help="The trained model to be used for forecasting",
    ),
    data_file: str = typer.Option(
        ...,
        "--data-file",
        "-df",
        help="The data file to be used for training the model",
    ),
    output_dir: str = typer.Option(
        ...,
        "--output-dir",
        "-o",
        help="The output directory where the forecasted data will be saved",
    ),
    forecast_method: ForecastMethod = typer.Option(
        "classic",
        "--forecast-method",
        "-fm",
        help="The method to be used for forecasting. The default is Classic.",
    ),
    forecast_length: int = typer.Option(
        1000,
        "--forecast-length",
        "-fl",
        help="The number of points to be forecasted. The default is 1000.",
    ),
    section_initialization_length: int = typer.Option(
        50,
        "--section-initialization-length",
        "-sil",
        help="The number of points to be used for initializing the sections with true data. The default is 50.",
    ),
    number_of_sections: int = typer.Option(
        10,
        "--number-of-sections",
        "-nos",
        help="The number of sections to be used for forecasting. The default is 10.",
    ),
):
    """Make predictions with a given model on a data file."""
    _forecast_from_saved_model(**locals())


# FIX: Output path
@app.command()
def plot(
    plot_type: PlotType = typer.Option(
        "linear",
        "--plot-type",
        "-pt",
        help="The type of plot to be made. The default is linear.",
    ),
    predictions: str = typer.Option(
        ...,
        "--predictions",
        "-pr",
        help="The path to the file containing the predictions.",
    ),
    data_file: str = typer.Option(
        ...,
        "--data_file",
        "-df",
        help="The path to the file containing the data.",
    ),
    lyapunov_exponent: float = typer.Option(
        1,
        "--lyapunov-exponent",
        "-le",
        help="The lyapunov exponent of the data. The default is 1. It is used to scale the time to the lyapunov time units.",
    ),
    delta_time: float = typer.Option(
        1,
        "--delta-time",
        "-dt",
        help="The time step between each point in the data. The default is 1.",
    ),
    plot_points: int = typer.Option(
        None,
        "--plot-points",
        "-pp",
        help="The number of points/steps to plot.",
    ),
    title: str = typer.Option(
        "",
        "--title",
        "-T",
        help="The title of the plot. The default is empty.",
    ),
    save_path: str = typer.Option(
        ".",
        "--save-path",
        "-sp",
        help="The path where the plot will be saved. The default is None.",
    ),
    show: bool = typer.Option(
        False,
        "--show/--no-show",
        "-s/-ns",
        help="Whether to show the plot or not. The default is True.",
    ),
    y_labels: str = typer.Option(
        None,
        "--y-labels",
        "-yl",
        help="The labels of the y axis. The default is None. Value should be a string with the labels separated by commas.",
    ),
    y_values: str = typer.Option(
        None,
        "--y-values",
        "-yv",
        help="The values of the y axis. The default is None. Value should be a string with both values separated by commas.",
    ),
    x_label: str = typer.Option(
        None,
        "--x-label",
        "-xl",
        help="The label of the x axis. The default is time (t).",
    ),
    transient: int = typer.Option(
        1000,
        "--transient",
        "-tr",
        help="The number of transient points discarded in the training of the model.",
    ),
    train_length: int = typer.Option(
        10000,
        "--train-length",
        "-tl",
        help="The number of points used for the training of the model.",
    ),
):
    """Plot different data."""
    _plot(**locals())


@app.command(help='Commad executed by the slurm grid search.')
def slurm_grid(
    data_path: str = typer.Option(..., "--data", "-d", help='Path to the system data.'),
    output_path: str = typer.Option(..., "--output", "-o", help='Output path.'),
    index: int = typer.Option(..., "--index", "-i", help='Index of the combination in the combinations.json file.'),
    hyperparameters_path: str = typer.Option(..., "--hyperparameters-path", "-hp", help='Path to the .json file of the combinations.'),
):
    _slurm_grid(**locals())


# INITIALIZE GRID
@app.command(help='Initialize all files and folders for grid search.')
def init_slurm_grid(
    path: str = typer.Option(..., '--path', '-p', help='Base path to save grid search folders and files.'),
    job_name: str = typer.Option('job', '--job-name', '-j', help='Slurm job name.'),
    data_path: str = typer.Option(..., '--data-path', '-dp', help='Path of the System data.'),

    model: str = typer.Option('ESN', '-m', '--model'),
    input_initializer: str = typer.Option('InputMatrix', '-ii', '--input-initializer'),
    input_bias_initializer: str = typer.Option('RandomUniform', '-ib', '--input-bias'),
    reservoir_activation: str = typer.Option('tanh', '-ra', '--reservoir-activation'),
    reservoir_initializer: str = typer.Option('WattsStrogatzNX', '-ri', '--reservoir-initializer'),

    units: List[int] = typer.Option([5000], '--units', '-u'),
    train_length: List[int] = typer.Option([20000], '--train-length', '-tl'),
    forecast_length: List[int] = typer.Option([1000], '--forecast-length', '-fl'),
    transient: List[int] = typer.Option([1000], '--transient', '-t'),
    steps: List[int] = typer.Option([1], '--steps', '-s'),

    input_scaling: List[float] = typer.Option(..., '--input-scaling', '-is'),
    leak_rate: List[float] = typer.Option(..., '--leak-rate', '-lr'),
    spectral_radius: List[float] = typer.Option(..., '--spectral-radius', '-sr'),
    rewiring: List[float] = typer.Option(..., '--rewiring', '-rw'),
    reservoir_degree: List[int] = typer.Option(..., '--reservoir-degree', '-rd'),
    reservoir_sigma: List[float] = typer.Option(..., '--reservoir-sigma', '-rs'),
    regularization: List[float] = typer.Option(..., '--regularization', '-rg'),
):
    _init_slurm_grid(**locals())


# FIX
# RUN BETWEEN GRID SEARCH
@app.command(help='Generate the next steps of the grid search from the results of the previous ones.')
def grid_aux(
    path: str = typer.Option(..., '--path', '-p', help='Base path to grid search folders'),
    n_results: int = typer.Option(..., "--n-results", "-nr"),
    threshold: float = typer.Option(..., "--threshold", "-t"),
):
    # TODO: Adapt method to new changes
    raise NotImplementedError
    _grid_aux(**locals())


@app.command(help='Get the best results from the given path. Compare by the given `threshold`.')
def best_results(
    results_path: str = typer.Option(..., "--results-path", "-rp"),
    output: str = typer.Option(..., "--output", "-o"),
    n_results: int = typer.Option(..., "--n-results", "-nr"),
    threshold: float = typer.Option(..., "--threshold", "-t"),
):
    _best_results(**locals())


@app.command(help='Generate the a .json file with the hyperparameters of every training and the index where the rmse from the results are bigger than the threshold.')
def results_data(
    results_path: str = typer.Option(..., "--results-path", "-rp", help='Path of the results from grid search to be analized.'),
    filepath: str = typer.Option(..., "--filepath", "-fp", help='File path for the output. Must be a .json file.'),
    threshold: float = typer.Option(..., "--threshold", "-t"),
):
    _results_data(**locals())
    

@app.command(help='Search for the combinations that have not been satisfactorily completed and create a script to execute them')
def search_unfinished_combinations(
    data_path: str = typer.Option(..., "--data-path", "-dp"),
    path:str =  typer.Option(..., "--path", "-p", help='Specify the folder where the results of the combinations are stored'),
    depth = typer.Option(0, "--depth", "-d", help='Grid depth, to specify the depth of the grid seach.')
):
    # TODO: Adapt method to new changes
    raise NotImplementedError
    _search_unfinished_combinations(**locals())


if __name__ == "__main__":
    app()
