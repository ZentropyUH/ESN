import os
from typer import Typer
from typer import Option

from cli.enums import *

app = Typer(
    help="",
    no_args_is_help=True,
)


@app.command(
    name="ESN",
    no_args_is_help=True,
)
def esn(
    data_file: str = Option(
        ...,
        "--data-file",
        "-df",
        help="Data file to be used for training and forecasting.",
    ),
    delta_time: float = Option(
        1,
        '--delta-time', '-dt'
    ),
    lyapunov_exponent: float = Option(
        1,
        '--lyapunov-exponent', '-ly'
    ),
    output_dir: str = Option(
        ...,
        "--output-dir",
        "-o",
        help="The directory where the results will be saved. The default is the current directory.",
    ),
    units: int = Option(
        ..., 
        "--units", 
        "-u", 
        help="Number of units of the reservoir. In case of Parallel_ESN, it is the number of units of each reservoir."
    ),
    input_initializer: EnumInputInitializer = Option(
        "InputMatrix",
        "--input-initializer",
        "-ii",
        help="The initializer for the input weights. The default is InputMatrix.",
        autocompletion=EnumInputInitializer.list,
    ),
    input_bias_initializer: EnumInputBiasInitializer = Option(
        "RandomUniform",
        "--input-bias-initializer",
        "-ib",
        help="The initializer for the input bias weights. The default is RandomUniform.",
        autocompletion=EnumInputBiasInitializer.list,
    ),
    input_scaling: float = Option(
        0.5,
        "--input-scaling",
        "-is",
        help="The input scaling parameter. The default is 0.5.",
    ),
    leak_rate: float = Option(
        1.0,
        "--leak-rate",
        "-lr",
        help="The leak rate of the reservoir. The default is 1.",
    ),
    reservoir_activation: EnumReservoirActivation = Option(
        "tanh",
        "--reservoir-activation",
        "-a",
        help="The activation function of the reservoir. The default is tanh. Only used if ESN or Parallel_ESN is used.",
        autocompletion=EnumReservoirActivation.list,
    ),
    seed: int = Option(
        None,
        "--seed",
        "-s",
        help="The seed to be used for the random number generator. If not specified, it is randomly generated.",
    ),
    spectral_radius: float = Option(
        0.99,
        "--spectral-radius",
        "-sr",
        help="The spectral radius of the reservoir. The default is 0.99.",
    ),
    reservoir_initializer: EnumReservoirInitializer = Option(
        "WattsStrogatzNX",
        "--reservoir-initializer",
        "-ri",
        help="The initializer for the reservoir weights. The default is WattsStrogatzNX. Only used if ESN or Parallel_ESN is used.",  # Maybe play later with topologies on ECA and Oscillators. First we have to study impact on EOC
        autocompletion=EnumReservoirInitializer.list,
    ),
    rewiring: float = Option(
        0.5,
        "--rewiring",
        "-rw",
        help="The rewiring probability of the WattsStrogatz graph. The default is 0.5. Only used if ESN or Parallel_ESN is used.",
    ),
    reservoir_degree: int = Option(
        3,
        "--reservoir-degree",
        "-rd",
        help="The degree of the reservoir. The default is 3. Only used if ESN or Parallel_ESN is used.",
    ),
    reservoir_sigma: float = Option(
        0.5,
        "--reservoir-sigma",
        "-rs",
        help="The standard deviation for the reservoir weights. The default is 0.5. Only used if ESN or Parallel_ESN is used.",
    ),
    readout_layer: EnumReadoutLayer = Option(
        "linear",
        "--readout-layer",
        "-rl",
        help="The type of readout layer of the model; 'linear' if the layer is a linear regression using Ridge (Tikhonov) regularization scheme; 'sgd' if the readout should be a linear regression to be calculated iteratively with stochastic gradient descent; 'mlp' if the readout is to be chosen as a multilayer perceptron. If 'mlp' is chosen more options should be provided.",
        autocompletion=EnumReadoutLayer.list,
    ),
    regularization: float = Option(
        1e-4,
        "--regularization",
        "-rg",
        help="The regularization parameter. The default is 1e-4.",
    ),
    transient: int = Option(
        1000,
        "--transient",
        "-tr",
        help="The number of transient points to be discarded. The default is 1000.",
    ),
    train_length: int = Option(
        10000,
        "--train-length",
        "-tl",
        help="The number of points to be used for training. The default is 10000.",
    ),
    steps: int = Option(
        1,
        "--steps",
        "-st",
        help="Number of steps among data point to ignore. Used to variate the data dt.",
    ),
    times: int = Option(
        1,
        "--times",
        "-t",
        help="Number of times to run the model.",
    ),
    forecast_method: EnumForecastMethod = Option(
        "classic",
        "--forecast-method",
        "-fm",
        help="The method to be used for forecasting. The default is Classic.",
        autocompletion=EnumForecastMethod.list,
    ),
    forecast_length: int = Option(
        1000,
        "--forecast-length",
        "-fl",
        help="The number of points to be forecasted. The default is 1000.",
    ),
):
    from research.grid.grid import grid

    for i in range(times):
        model_path = os.path.join(output_dir, str(i))

        grid(
            data_path=data_file,
            output_path=model_path,
            units=units,
            train_length=train_length,
            forecast_length=forecast_length,
            transient=transient,
            steps=steps,
            dt=delta_time,
            lyapunov_exponent=lyapunov_exponent,
            model=EnumModel.ESN.value,
            input_initializer=input_initializer,
            input_bias_initializer=input_bias_initializer,
            reservoir_activation=reservoir_activation,
            reservoir_initializer=reservoir_initializer,
            input_scaling=input_scaling,
            leak_rate=leak_rate,
            spectral_radius=spectral_radius,
            rewiring=rewiring,
            reservoir_degree=reservoir_degree,
            reservoir_sigma=reservoir_sigma,
            regularization=regularization,
        )


# @app.command(
#     name="Parallel-ESN",
#     no_args_is_help=True,
# )
# def parallel_esn(
#     data_file: str = Option(
#         ...,
#         "--data-file",
#         "-df",
#         help="Data file to be used for training and forecasting.",
#     ),
#     delta_time: float = Option(
#         1,
#         '--delta-time', '-dt'
#     ),
#     lyapunov_exponent: float = Option(
#         1,
#         '--lyapunov-exponent', '-ly'
#     ),
#     output_dir: str = Option(
#         ...,
#         "--output-dir",
#         "-o",
#         help="The directory where the results will be saved. The default is the current directory.",
#     ),
#     units: int = Option(
#         ..., 
#         "--units", 
#         "-u", 
#         help="Number of units of the reservoir. In case of Parallel_ESN, it is the number of units of each reservoir."
#     ),
#     input_initializer: EnumInputInitializer = Option(
#         "InputMatrix",
#         "--input-initializer",
#         "-ii",
#         help="The initializer for the input weights. The default is InputMatrix.",
#         autocompletion=EnumInputInitializer.list,
#     ),
#     input_bias_initializer: EnumInputBiasInitializer = Option(
#         "RandomUniform",
#         "--input-bias-initializer",
#         "-ib",
#         help="The initializer for the input bias weights. The default is RandomUniform.",
#         autocompletion=EnumInputBiasInitializer.list,
#     ),
#     input_scaling: float = Option(
#         0.5,
#         "--input-scaling",
#         "-is",
#         help="The input scaling parameter. The default is 0.5.",
#     ),
#     leak_rate: float = Option(
#         1.0,
#         "--leak-rate",
#         "-lr",
#         help="The leak rate of the reservoir. The default is 1.",
#     ),
#     reservoir_activation: EnumReservoirActivation = Option(
#         "tanh",
#         "--reservoir-activation",
#         "-a",
#         help="The activation function of the reservoir. The default is tanh. Only used if ESN or Parallel_ESN is used.",
#         autocompletion=EnumReservoirActivation.list,
#     ),
#     seed: int = Option(
#         None,
#         "--seed",
#         "-s",
#         help="The seed to be used for the random number generator. If not specified, it is randomly generated.",
#     ),
#     spectral_radius: float = Option(
#         0.99,
#         "--spectral-radius",
#         "-sr",
#         help="The spectral radius of the reservoir. The default is 0.99.",
#     ),
#     reservoir_initializer: EnumReservoirInitializer = Option(
#         "WattsStrogatzNX",
#         "--reservoir-initializer",
#         "-ri",
#         help="The initializer for the reservoir weights. The default is WattsStrogatzNX. Only used if ESN or Parallel_ESN is used.",  # Maybe play later with topologies on ECA and Oscillators. First we have to study impact on EOC
#         autocompletion=EnumReservoirInitializer.list,
#     ),
#     rewiring: float = Option(
#         0.5,
#         "--rewiring",
#         "-rw",
#         help="The rewiring probability of the WattsStrogatz graph. The default is 0.5. Only used if ESN or Parallel_ESN is used.",
#     ),
#     reservoir_degree: int = Option(
#         3,
#         "--reservoir-degree",
#         "-rd",
#         help="The degree of the reservoir. The default is 3. Only used if ESN or Parallel_ESN is used.",
#     ),
#     reservoir_sigma: float = Option(
#         0.5,
#         "--reservoir-sigma",
#         "-rs",
#         help="The standard deviation for the reservoir weights. The default is 0.5. Only used if ESN or Parallel_ESN is used.",
#     ),
#     readout_layer: EnumReadoutLayer = Option(
#         "linear",
#         "--readout-layer",
#         "-rl",
#         help="The type of readout layer of the model; 'linear' if the layer is a linear regression using Ridge (Tikhonov) regularization scheme; 'sgd' if the readout should be a linear regression to be calculated iteratively with stochastic gradient descent; 'mlp' if the readout is to be chosen as a multilayer perceptron. If 'mlp' is chosen more options should be provided.",
#         autocompletion=EnumReadoutLayer.list,
#     ),
#     regularization: float = Option(
#         1e-4,
#         "--regularization",
#         "-rg",
#         help="The regularization parameter. The default is 1e-4.",
#     ),
#     reservoir_amount: int = Option(
#         1,
#         "--reservoir-amount",
#         "-ra",
#         help="The number of reservoirs to be used. The default is 10. Only used if Parallel_ESN is used or other parallel scheme.",
#     ),
#     overlap: int = Option(
#         0,
#         "--overlap",
#         "-ol",
#         help="The number of overlapping units between reservoirs. The default is 6. Only used if Parallel_ESN is used.",
#     ),
#     transient: int = Option(
#         1000,
#         "--transient",
#         "-tr",
#         help="The number of transient points to be discarded. The default is 1000.",
#     ),
#     train_length: int = Option(
#         10000,
#         "--train-length",
#         "-tl",
#         help="The number of points to be used for training. The default is 10000.",
#     ),
#     steps: int = Option(
#         1,
#         "--steps",
#         "-s",
#         help="Number of steps among data point to ignore. Used to variate the data dt.",
#     ),
#     times: int = Option(
#         1,
#         "--times",
#         "-t",
#         help="Number of times to run the model.",
#     ),
#     forecast_method: EnumForecastMethod = Option(
#         "classic",
#         "--forecast-method",
#         "-fm",
#         help="The method to be used for forecasting. The default is Classic.",
#         autocompletion=EnumForecastMethod.list,
#     ),
#     forecast_length: int = Option(
#         1000,
#         "--forecast-length",
#         "-fl",
#         help="The number of points to be forecasted. The default is 1000.",
#     ),
# ):
#     from research.grid.grid import grid

#     for i in range(times):
#         output_dir = os.path.join(output_dir, str(i))

#         grid(
#             data_path=data_file,
#             output_path=output_dir,
#             units=units,
#             train_length=train_length,
#             forecast_length=forecast_length,
#             transient=transient,
#             steps=steps,
#             dt=delta_time,
#             lyapunov_exponent=lyapunov_exponent,
#             model=EnumModel.ESN.value,
#             input_initializer=input_initializer,
#             input_bias_initializer=input_bias_initializer,
#             reservoir_activation=reservoir_activation,
#             reservoir_initializer=reservoir_initializer,
#             input_scaling=input_scaling,
#             leak_rate=leak_rate,
#             spectral_radius=spectral_radius,
#             rewiring=rewiring,
#             reservoir_degree=reservoir_degree,
#             reservoir_sigma=reservoir_sigma,
#             regularization=regularization,
#             reservoir_amount=reservoir_amount,
#             overlap=overlap,
#         )