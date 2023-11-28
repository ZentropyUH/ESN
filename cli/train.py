from typer import Typer
from typer import Option

from cli.enums import *


app = Typer(
    help="",
    no_args_is_help=True,
)


@app.command(
    name="esn",
    no_args_is_help=True,
)
def train_ESN(
    data_file: str = Option(
        ...,
        "--data-file",
        "-df",
        help="Data file to be used for training.",
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
        "-s",
        help="Number of steps among data point to ignore. Used to variate the data dt.",
    ),
):
    """Train a specific model on a given data file."""
    from functions import train
    train(
        data_file=data_file,
        output_dir=output_dir,
        model=EnumModel.ESN.value,
        units=units,
        input_initializer=input_initializer,
        input_bias_initializer=input_bias_initializer,
        input_scaling=input_scaling,
        leak_rate=leak_rate,
        reservoir_activation=reservoir_activation,
        seed=seed,
        spectral_radius=spectral_radius,
        reservoir_initializer=reservoir_initializer,
        rewiring=rewiring,
        reservoir_degree=reservoir_degree,
        reservoir_sigma=reservoir_sigma,
        readout_layer=readout_layer,
        regularization=regularization,
        transient=transient,
        train_length=train_length,
        steps=steps,
    )


@app.command(
    name="pesn",
    no_args_is_help=True,
)
def train_parallel_ESN(
    data_file: str = Option(
        ...,
        "--data-file",
        "-df",
        help="Data file to be used for training.",
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
    reservoir_amount: int = Option(
        1,
        "--reservoir-amount",
        "-ra",
        help="The number of reservoirs to be used. The default is 10. Only used if Parallel_ESN is used or other parallel scheme.",
    ),
    overlap: int = Option(
        0,
        "--overlap",
        "-ol",
        help="The number of overlapping units between reservoirs. The default is 6. Only used if Parallel_ESN is used.",
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
        "-s",
        help="Number of steps among data point to ignore. Used to variate the data dt.",
    ),
):
    """Train a specific model on a given data file."""
    from functions import train
    train(
        data_file=data_file,
        output_dir=output_dir,
        model=EnumModel.PESN.value,
        units=units,
        input_initializer=input_initializer,
        input_bias_initializer=input_bias_initializer,
        input_scaling=input_scaling,
        leak_rate=leak_rate,
        reservoir_activation=reservoir_activation,
        seed=seed,
        spectral_radius=spectral_radius,
        reservoir_initializer=reservoir_initializer,
        rewiring=rewiring,
        reservoir_degree=reservoir_degree,
        reservoir_sigma=reservoir_sigma,
        reservoir_amount=reservoir_amount,
        overlap=overlap,
        readout_layer=readout_layer,
        regularization=regularization,
        transient=transient,
        train_length=train_length,
        steps=steps,
    )


@app.command(
    name="eca",
    no_args_is_help=True,
)
def train_eca_ESN(
        data_file: str = Option(
        ...,
        "--data-file",
        "-df",
        help="Data file to be used for training.",
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
        "-s",
        help="Number of steps among data point to ignore. Used to variate the data dt.",
    ),
    eca_rule: int = Option(
        110,
        "--eca-rule",
        "-er",
        help="The rule to be used for the ECA automaton. The default is 110.",
    ),
    eca_steps: int = Option(
        1,
        "--eca-steps",
        "-es",
        help="Number of steps for the ECA automaton.",
    ),
):
    """Train a specific model on a given data file."""
    from functions import train
    train(
        data_file=data_file,
        output_dir=output_dir,
        model=EnumModel.ECA.value,
        units=units,
        input_initializer=input_initializer,
        input_bias_initializer=input_bias_initializer,
        input_scaling=input_scaling,
        leak_rate=leak_rate,
        reservoir_activation=reservoir_activation,
        seed=seed,
        readout_layer=readout_layer,
        regularization=regularization,
        transient=transient,
        train_length=train_length,
        steps=steps,
        eca_rule=eca_rule,
        eca_steps=eca_steps,
    )

if __name__ == "__main__":
    app()
