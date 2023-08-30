#!/usr/bin/python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import typer

from src.grid.tools import *
from src.utils import load_model_and_params

from t_utils import *
from functions import _train, _forecast, _plot
from src.grid.grid import _grid
from src.grid.tools import get_best_results, generate_result_combinations, script_generator, generate_initial_combinations

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
    file_name: str = typer.Option(
        None,
        "--file-name",
        "-fn",
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
        "WattsStrogatzOwn",
        "--reservoir-initializer",
        "-ri",
        help="The initializer for the reservoir weights. The default is WattsStrogatzOwn. Only used if ESN or Parallel_ESN is used.",  # Maybe play later with topologies on ECA and Oscillators. First we have to study impact on EOC.allow_from_autoenv=
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
    forecast_params = locals()
    trained_model, model_params = load_model_and_params(trained_model_path)
    forecast_params.pop("trained_model_path")
    print(forecast_params)
    _forecast(trained_model, model_params, **forecast_params)


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


@app.command()
def grid(
    units: int = typer.Option(9000, "--units", "-u"),
    train_length: int = typer.Option(20000, "--train-length", "-tl"),
    forecast_length: int = typer.Option(1000, "--forecast-length", "-fl"),
    transient: int = typer.Option(1000, "--transient", "-tr"),
    steps: int = typer.Option(1, '--steps', '-s'),

    data_path: str = typer.Option(..., "--data", "-d"),
    output_path: str = typer.Option(..., "--output", "-o"),
    
    index: int = typer.Option(..., "--index", "-i"),
    hyperparameters_path: str = typer.Option(..., "--hyperparameters-path", "-hp"),
):
    _grid(
        units=units,
        train_length=train_length,
        forecast_length=forecast_length,
        transient=transient,
        steps=steps,
        data_path=data_path,
        output_path=output_path,
        index=index,
        hyperparameters_path=hyperparameters_path,
    )



# INITIALIZE GRID
@app.command()
def grid_init(
    path: str = typer.Option(..., "--path", "-p"),
    job_name: str = typer.Option(..., "--job-name", "-j"),
    data_path: str = typer.Option(..., "--data-path", "-dp"),
    steps: int = typer.Option(1, '--steps', '-s'),
):
    info_path = join(path, 'info')
    makedirs(info_path, exist_ok=True)
    n_info_path = join(info_path, '0')
    makedirs(n_info_path, exist_ok=True)
    n_run_path = join(path, 'run_0')
    makedirs(n_run_path, exist_ok=True)
    combinations_path = join(n_info_path, 'combinations.json')
    output_path = join(n_run_path, 'data')
    script_file = join(n_info_path, 'script.sh')

    combinations = generate_initial_combinations(n_info_path)
    script_generator(
        job_name,
        (1, len(combinations)),
        combinations_path,
        output_path,
        data_path,
        script_file,
        steps
    )



@app.command(help='Generate and save the initial hyperparameters combinations in the given path.')
def initial_combinations(
    output: str = typer.Option(..., "--output", "-o"),
):
    generate_initial_combinations(output)


@app.command(help='Generate new slurm script.')
def script(
    job_name: str = typer.Option(..., "--job-name", "-j"),
    data_path: str = typer.Option(..., "--data-path", "-dp"),
    combinations_path: str = typer.Option(..., "--combinations-path", "-cp"),
    output_path: str = typer.Option(..., "--output-path", "-op"),
    filepath: str = typer.Option(..., "--file-path", "-fp"),
    steps: int = typer.Option(1, '--steps', '-s'),
):
    combinations = load_hyperparams(combinations_path)
    script_generator(
        job_name,
        (1, len(combinations)),
        combinations_path,
        output_path,
        data_path,
        filepath,
        steps
    )



# RUN BETWEEN GRID SEARCH
@app.command()
def grid_aux(
    job_name: str = typer.Option(..., "--job-name", "-j"),
    run_path: str = typer.Option(..., "--run-path", "-rp"),
    data_path: str = typer.Option(..., "--data-path", "-dp"),
    info_path: str = typer.Option(..., "--info-path", "-ip"),
    n_results: int = typer.Option(..., "--n-results", "-nr"),
    threshold: float = typer.Option(..., "--threshold", "-t"),
    steps: int = typer.Option(1, '--steps', '-s'),
):
    output_path = join(run_path, 'data')
    results_path = join(run_path, 'results')
    steps_file = join(info_path, 'steps.json')
    new_info = join(Path(info_path).absolute().parent, str(int(Path(info_path).absolute().name)+1))
    makedirs(new_info, exist_ok=True)
    new_run = join(Path(run_path).absolute().parent, 'run_' + str(int(Path(run_path).absolute().name.split('_')[-1])+1))
    makedirs(new_run, exist_ok=True)

    best_results(
        output_path,
        results_path,
        n_results,
        threshold
    )
    new_combinations = generate_result_combinations(
        results_path,
        steps_file,
        new_info,
    )
    script_generator(
        job_name,
        (1, len(new_combinations)),
        join(new_info, 'combinations.json'),
        join(new_run, 'data'),
        data_path,
        join(new_info, 'script.sh'),
        steps
    )


@app.command(help='Generate the new hyperparameters combinations from the results from the given path.')
def new_combinations(
    path: str = typer.Option(..., "--path", "-p"),
    output: str = typer.Option(..., "--output", "-o"),
    steps: str = typer.Option(..., "--steps", "-s"),
):
    generate_result_combinations(
        path = path,
        steps_file = steps,
        output = output,
    )


@app.command(help='Get the best results from the given path. Compare by the given `threshold`.')
def best_results(
    path: str = typer.Option(..., "--path", "-p"),
    output: str = typer.Option(..., "--output", "-o"),
    max_size: int = typer.Option(..., "--max-size", "-ms"),
    threshold: float = typer.Option(..., "--threshold", "-t"),
):
    get_best_results(
        path,
        output,
        max_size,
        threshold
    )



@app.command()
def results_data(
    path: str = typer.Option(..., "--path", "-p"),
    filepath: str = typer.Option(..., "--file-path", "-fp"),
    threshold: float = typer.Option(..., "--threshold", "-t"),
):
    results_info(path, filepath, threshold)
    


if __name__ == "__main__":
    app()
