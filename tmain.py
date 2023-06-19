import typer
from src.grid.grid_tools import *

from t_utils import *
from src.grid.grid_one import grid_one, best_combinations, change_folders, detect_not_fished_jobs, calculate_mse
from model_functions import _train, _forecast, _plot


app = typer.Typer()

#TODO: Files names



@app.command()
def train(
    # General params
    model: EModel = typer.Option(
        'ESN',
        '--model', '-m',
        help=''
    ),
    units: str = typer.Option(
        ...,
        '--units', '-u',
        help=''
    ),
    input_initializer: EInputInitializer = typer.Option(
        'InputMatrix',
        '--input-initializer', '-ii',
        help= '',
    ),
    input_bias_initializer: InputBiasInitializer = typer.Option(
        'RandomUniform',
        '--input-bias-initializer', '-ib',
        help="The initializer for the input bias weights. The default is RandomUniform.",
    ),
    input_scaling: str = typer.Option(
        '0.5',
        '--input-scaling', '-is',
        help="The input scaling parameter. The default is 0.5. If a range of values is given, the script will be executed the specified number of times with different values of the input scaling parameter. The values will be chosen linearly between the first and the second value. If a list of values is given, the script will be executed the specified number of times with the values in the list.",
    ),
    leak_rate: str = typer.Option(
        '1.',
        '--leak-rate', '-lr',
        help="The leak rate of the reservoir. The default is 1. If a range of values is given, the script will be executed the specified number of times with different values of the leak rate. The values will be chosen linearly between the first and the second value. If a list of values is given, the script will be executed the specified number of times with the values in the list.",
    ),
    reservoir_activation: ReservoirActivation = typer.Option(
        'tanh',
        '--reservoir-activation', '-a',
        help="The activation function of the reservoir. The default is tanh. Only used if ESN or Parallel_ESN is used.",
    ),
    # Classic Cases
    spectral_radius: str = typer.Option(
        '0.99',
        '--spectral-radius', '-sr',
        help="The spectral radius of the reservoir. The default is 0.99. Only used if ESN or Parallel_ESN is used. If a range of values is given, the script will be executed the specified number of times with different values of the spectral radius. The values will be chosen linearly between the first and the second value. If a list of values is given, the script will be executed the specified number of times with the values in the list.",
    ),
    reservoir_initializer: ReservoirInitializer = typer.Option(
        'WattsStrogatzOwn',
        '--reservoir-initializer', '-ri',
        help="The initializer for the reservoir weights. The default is WattsStrogatzOwn. Only used if ESN or Parallel_ESN is used.",  # Maybe play later with topologies on ECA and Oscillators. First we have to study impact on EOC.allow_from_autoenv=
    ),
    rewiring: str = typer.Option(
        '0.5',
        '--rewiring', '-rw',
        help="The rewiring probability of the WattsStrogatz graph. The default is 0.5. Only used if ESN or Parallel_ESN is used. If a range of values is given, the script will be executed the specified number of times with different values of the degree parameter. The values will be chosen linearly between the first and the second value. If a list of values is given, the script will be executed the specified number of times with the values in the list.",
    ),
    reservoir_degree: str = typer.Option(
        '3',
        '--reservoir-degree', '-rd',
        help="The degree of the reservoir. The default is 3. Only used if ESN or Parallel_ESN is used. If a range of values is given, the script will be executed the specified number of times with different values of the degree parameter. The values will be chosen linearly between the first and the second value. If a list of values is given, the script will be executed the specified number of times with the values in the list.",
    ),
    reservoir_sigma: str = typer.Option(
        '0.5',
        '--reservoir-sigma', '-rs',
        help="The standard deviation for the reservoir weights. The default is 0.5. Only used if ESN or Parallel_ESN is used. If a range of values is given, the script will be executed the specified number of times with different values of the sigma parameter. The values will be chosen linearly between the first and the second value. If a list of values is given, the script will be executed the specified number of times with the values in the list.",
    ),
    # Parallel cases
    reservoir_amount: int = typer.Option(
        10,
        '--reservoir-amount', '-ra',
        help="The number of reservoirs to be used. The default is 10. Only used if Parallel_ESN is used or other parallel scheme.",
    ),
    overlap: str = typer.Option(
        '6',
        "--overlap", "-ol",
        help="The number of overlapping units between reservoirs. The default is 6. Only used if Parallel_ESN is used or other parallel scheme. If a range of values is given, the script will be executed the specified number of times with different values of the overlap parameter. The values will be chosen linearly between the first and the second value. If a list of values is given, the script will be executed the specified number of times with the values in the list.",
    ),
    # Readout params
    readout_layer: ReadoutLayer = typer.Option(
        'linear',
        "--readout-layer", "-rl",
        help="The type of readout layer of the model; 'linear' if the layer is a linear regression using Ridge (Tikhonov) regularization scheme; 'sgd' if the readout should be a linear regression to be calculated iteratively with stochastic gradient descent; 'mlp' if the readout is to be chosen as a multilayer perceptron. If 'mlp' is chosen more options should be provided.",
    ),
    regularization: str = typer.Option(
        '1e-4',
        "--regularization", "-rg",
        help="The regularization parameter. The default is 1e-4. If a range of values is given, the script will be executed the specified number of times with different values of the regularization parameter. The values will be chosen logarithmically between the first and the second value. If a list of values is given, the script will be executed the specified number of times with the values in the list.",
    ),
    # Training params
    init_transient: int = typer.Option(
        1000,
        "--init-transient", "-it",
        help="The number of transient points to be discarded. The default is 1000.",
    ),
    transient: int = typer.Option(
        1000,
        "--transient", "-tr",
        help="The number of transient points to be discarded. The default is 1000.",
    ),
    train_length: str = typer.Option(
        '10000',
        "--train-length", "-tl",
        help="The number of points to be used for training. The default is 10000. If a range of values is given, the script will be executed the specified number of times worg.freedesktop.PackageKit.proxyith different values of the training length. The values will be chosen linearly between the first and the second value. If a list of values is given, the script will be executed the specified number of times with the values in the list.",
    ),
    data_file: str = typer.Option(
        ...,
        "--data-file", "-df",
        help="Data file to be used for training.",
    ),
    output_dir: str = typer.Option(
        ...,
        "--output-dir", "-o",
        help="The directory where the results will be saved. The default is the current directory.",
    ),
    file_name: str = typer.Option(
        ...,
        "--file-name", "-fn",
    ),
):
    '''Train a specific model on a given data file.'''
    _train(**locals())




@app.command()
def forecast(
    forecast_method: ForecastMethod = typer.Option(
        'classic',
        "--forecast-method", "-fm",
        help="The method to be used for forecasting. The default is Classic.",
    ),
    forecast_length: int = typer.Option(
        1000,
        "--forecast-length", "-fl",
        help="The number of points to be forecasted. The default is 1000.",
    ),
    section_initialization_length: int = typer.Option(
        50,
        "--section-initialization-length", "-sil",
        help="The number of points to be used for initializing the sections with true data. The default is 50.",
    ),
    number_of_sections: int = typer.Option(
        10,
        "--number-of-sections", "-nos",
        help="The number of sections to be used for forecasting. The default is 10.",
    ),
    output_dir: str = typer.Option(
        ...,
        "--output-dir", "-o",
        help="The output directory where the forecasted data will be saved",
    ),
    trained_model: str = typer.Option(
        ...,
        "--trained-model", "-tm",
        help="The trained model to be used for forecasting",
    ),
    data_file: str = typer.Option(
        ...,
        "--data-file", "-df",
        help="The data file to be used for training the model",
    ),
    file_name: str = typer.Option(
        ...,
        "--file-name", "-fn",
    ),
):
    '''Make predictions with a given model on a data file.'''
    _forecast(**locals())




#FIX: Output path
@app.command()
def plot(
    plot_type: PlotType = typer.Option(
        'linear',
        "--plot-type", "-pt",
        help="The type of plot to be made. The default is linear.",
    ),
    predictions: str = typer.Option(
        ...,
        "--predictions", "-pr",
        help="The path to the file containing the predictions.",
    ),
    data_file: str = typer.Option(
        ...,
        "--data_file", "-df",
        help="The path to the file containing the data.",
    ),
    lyapunov_exponent: float = typer.Option(
        1,
        "--lyapunov-exponent", "-le",
        help="The lyapunov exponent of the data. The default is 1. It is used to scale the time to the lyapunov time units.",
    ),
    delta_time: float = typer.Option(
        1,
        "--delta-time", "-dt",
        help="The time step between each point in the data. The default is 1.",
    ),
    plot_points: int = typer.Option(
        None,
        "--plot-points", "-pp",
        help="The number of points/steps to plot."
    ),
    title: str = typer.Option(
        '',
        "--title", "-T",
        help="The title of the plot. The default is empty.",
    ),
    save_path: str = typer.Option(
        '.',
        "--save-path", "-sp",
        help="The path where the plot will be saved. The default is None.",
    ),
    show: bool = typer.Option(
        False,
        "--show/--no-show", "-s/-ns",
        help="Whether to show the plot or not. The default is True.",
    ),
    y_labels: str = typer.Option(
        None,
        "--y-labels", "-yl",
        help="The labels of the y axis. The default is None. Value should be a string with the labels separated by commas.",
    ),
    y_values: str = typer.Option(
        None,
        "--y-values", "-yv",
        help="The values of the y axis. The default is None. Value should be a string with both values separated by commas.",
    ),
    x_label: str = typer.Option(
        None,
        "--x-label", "-xl",
        help="The label of the x axis. The default is time (t).",
    ),
    init_transient: int = typer.Option(
        1000,
        "--init-transient", "-it",
        help="The number of transient points that were discarded at the beginning of the data.",
    ),
    transient: int = typer.Option(
        1000,
        "--transient", "-tr",
        help="The number of transient points discarded in the training of the model.",
    ),
    train_length: int = typer.Option(
        10000,
        "--train-length", "-tl",
        help="The number of points used for the training of the model.",
    ),
):
    '''Plot different data.'''
    _plot(**locals())




@app.command()
def grid(
    index: int = typer.Option(..., '--index', '-i'),
    data_path: str = typer.Option(..., '--data', '-d'),
    output_path: str = typer.Option(..., '--output', '-o'),
    units: int = typer.Option(9000, '--units', '-u'),
    training_lenght: int = typer.Option(20000, '--training-lenght', '-tl'),
):
    grid_one(
        index,
        data_path,
        output_path,
        units,
        training_lenght,
    )




@app.command()
def best_params(
    path: str = typer.Option(..., '--path', '-p'),
    output: str = typer.Option(..., '--output', '-o'),
    max: int = typer.Option(..., '--max', '-m'),
    threshold: float = typer.Option(..., '--threshold', '-t'),
):
    best_combinations(
        path,
        output,
        max,
        threshold,
    )




@app.command()
def cf(
    path: str = typer.Option(..., '--path', '-p'),
):
    change_folders(
        path,
    )




@app.command()
def dnfj(
    path: str = typer.Option(..., '--path', '-p'),
    output: str = typer.Option(..., '--output', '-o'),
):
    detect_not_fished_jobs(
        path,
        output,
    )




@app.command()
def mse(
    path: str = typer.Option(..., '--path', '-p'),
    data_path: str = typer.Option(..., '--data', '-d'),
    output: str = typer.Option(..., '--output', '-o'),
    tl: int = typer.Option(..., '--training-lenght', '-tl'),
    trancient: int = typer.Option(..., '--trancient', '-t'),
):
    calculate_mse(
        path,
        data_path,
        output,
        tl,
        trancient,
    )




@app.command()
def plot_data_forecast(
    data_path: str = typer.Option(..., '--data', '-d'),
    forecast_path: str = typer.Option(..., '--forecast', '-f'),
    output: str = typer.Option(..., '--output', '-o'),
    trancient: int = typer.Option(..., '--trancient', '-t'),
):
    forecast_data = [join(forecast_path, x) for x in listdir(forecast_path)]
    data: list[str] = [join(data_path, p) for p in listdir(data_path)]

    for i, values in enumerate([(read_csv(d), read_csv(f)) for d, f in  zip(data, forecast_data)]):
        dd, ff = values
        dd = dd[trancient:]
        plots_data_forecast(dd[:1000], ff, join(output, str(i)))



@app.command()
def missing_cases(
    path: str = typer.Option(..., '--path', '-p'),
):
    l = []
    for i in listdir(path):
        if i.endswith('.out'):
            l.append(int(i.strip('slurm-').strip('.out').split('_')[1]))
    
    l.sort()
    print('list')
    print(l)
    print('missing')
    m = []
    for i in range(1, 9601):
        if i not in l:
            m.append(i)
    print(m)




def ranges(
    l: list[int],
):
    b = []
    c = [l[0]]
    for i in l[1:]:
        if i == c[-1] + 1:
            c.append(i)
        else:
            b.append(c)
            c = [i]
    
    print(','.join(['{}-{}'.format(x[0], x[-1]) if len(x) > 1 else str(x) for x in b]))


if __name__ == "__main__":
    app()