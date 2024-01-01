import os
# To eliminate tensorflow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json
import numpy as np
import pandas as pd
from os.path import join
from keras.initializers import Zeros
from keras.initializers import RandomUniform

from src.model import ESN
from src.model import generate_ESN
from src.model import generate_Parallel_ESN
from src.model import generate_ECA_ESN

from src.utils import load_data
from src.customs.custom_initializers import ErdosRenyi
from src.customs.custom_initializers import InputMatrix
from src.customs.custom_initializers import RegularNX
from src.customs.custom_initializers import WattsStrogatzNX


def train(
    data_file: str,
    model: str,
    units: int,
    transient: int,
    train_length: int,
    input_initializer: str,
    input_bias_initializer: str,
    input_scaling: float,
    leak_rate: float,
    reservoir_activation: str,

    regularization: float, #FIX
    # readout_layer: str = None, #FIX

    output_dir: str = None,
    seed: int | None = None,
    **kwargs,
):
    '''
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
    '''

    if seed is None:
        seed = np.random.randint(0, 100000000)
    params = locals().copy()
    args = params.pop('kwargs')
    params.update(args)

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
            input_initializer = RandomUniform(
                minval=-input_scaling,
                maxval=input_scaling,
            )

    ############### CHOOSE THE INPUT BIAS INITIALIZER ###############

    match input_bias_initializer:
        case "InputMatrix":
            input_bias_initializer = InputMatrix(sigma=input_scaling)
        case "RandomUniform":
            input_bias_initializer = RandomUniform(
                minval=-input_scaling,
                maxval=input_scaling,
            )

        case "None":
            input_bias_initializer = Zeros()

    ############### CHOOSE THE RESERVOIR INITIALIZER ###############

    reservoir_initializer = kwargs.get("reservoir_initializer", None)
    reservoir_degree = kwargs.get("reservoir_degree", 3)
    spectral_radius = kwargs.get("spectral_radius", 0.9)
    rewiring = kwargs.get("rewiring", 0.5)
    reservoir_sigma = kwargs.get("reservoir_sigma", 0.5)


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
        case "None":
            pass

    ############### CHOOSE THE MODEL ###############

    match model:
        case "ESN":
            
            _model = generate_ESN(
                units=units,
                leak_rate=leak_rate,
                features=features,
                activation=reservoir_activation,
                input_reservoir_init=input_initializer,
                input_bias_init=input_bias_initializer,
                reservoir_kernel_init=reservoir_initializer,
                exponent=2,
                seed=seed,
            )

        case "Parallel-ESN":
            overlap = kwargs.get("overlap", 0)
            reservoir_amount = kwargs.get("reservoir_amount", 1)
            _model = generate_Parallel_ESN(
                units=units,
                partitions=reservoir_amount,
                overlap=overlap,
                leak_rate=leak_rate,
                features=features,
                activation=reservoir_activation,
                input_reservoir_init=input_initializer,
                input_bias_init=input_bias_initializer,
                reservoir_kernel_init=reservoir_initializer,
                exponent=2,
                seed=seed,
            )

        case "ECA":
            
            rule = kwargs.get("eca_rule", 110)
            steps = kwargs.get("eca_steps", 1)
            
            _model = generate_ECA_ESN(
                units=units,
                rule=rule,
                steps=steps,
                leak_rate=leak_rate,
                features=features,
                activation=reservoir_activation,
                input_reservoir_init=input_initializer,
                input_bias_init=input_bias_initializer,
                exponent=2,
                seed=seed,
            )


    _model.train(
        transient_data,
        train_data,
        train_target,
        regularization
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        _model.save(output_dir)

        with open(join(output_dir, 'params.json'), 'w', encoding='utf-8') as f:
            json.dump(
                params,
                f,
                indent=4,
                sort_keys=True,
                separators=(",", ": ")
            )

    return _model


def forecast(
    trained_model: ESN,
    transient: int,
    train_length: int,
    data_file: str,
    output_dir: str = None,
    forecast_method: str = "classic",
    forecast_length: int = 1000,
    internal_states: bool = False,
    feedback_metrics: bool = True,
    **kwargs,
):
    '''
    Load a model and forecast the data.

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
        tuple: A tuple containing the predicted values and the true values.
    '''

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

    ############### CHOOSE THE FORECAST METHOD AND FORECAST ###############
    match forecast_method:
        case "classic":
            predictions, states_over_time = trained_model.forecast(
                forecast_length,
                forecast_transient_data,
                val_data,
                val_target,
                internal_states,
                feedback_metrics
            )

            predictions = predictions[0]

        case "section":
            raise NotImplementedError(f"{forecast_method} is yet to be implemented")


    # Save forecasted data
    if output_dir:
        output_file = os.path.join(output_dir, os.path.basename(data_file)) if os.path.isdir(output_dir) else output_dir
        pd.DataFrame(predictions).to_csv(output_file, index=False, header=None)

    # Handle internal states
    if internal_states:
        file_name_without_extension = os.path.splitext(os.path.basename(data_file))[0]
        internal_state_dir = os.path.join(os.path.dirname(output_dir), f"{file_name_without_extension}_internal_states")
        os.makedirs(internal_state_dir, exist_ok=True)
        internal_states_file = os.path.join(internal_state_dir, f"{file_name_without_extension}_states.csv")
        pd.DataFrame(states_over_time).to_csv(internal_states_file, index=False, header=None)

    return predictions, val_target[:, :forecast_length, :][0]


def forecast_from_saved_model(
    trained_model_path: str,
    data_file: str,
    forecast_method: str = "classic",
    forecast_length: int = 1000,
    output_dir: str = None,

    internal_states: bool = False,
    feedback_metrics: bool = True,
        
    **kwargs,
) -> None:
    '''
    Load a model and forecast the data.
    
    Args:
        trained_model_path (str): Path to the trained model
        data_file (str): The data file to be used for training the model
        output_dir (str): Path for save the forecasted data

        forecast_method (str): The method to be used for forecasting. The default is ClassicForecast.
        forecast_length (int): The number of points to be forecasted. The default is 1000.

        section_initialization_length: int = 50,
        number_of_sections: int = 10,
    '''
    with open(join(trained_model_path, 'params.json'), encoding="utf-8") as f:
        params = json.load(f)
    model = ESN.load(trained_model_path)

    forecast(
        trained_model=model,
        transient=params['transient'],
        train_length=params['train_length'],
        data_file=data_file,
        output_dir=output_dir,
        forecast_method=forecast_method,
        forecast_length=forecast_length,
        internal_states=internal_states,
        feedback_metrics=feedback_metrics
    )


def forecast_folder_from_saved_model(
    trained_model_path: str,
    data_folder: str,
    output_dir: str = None,
    
    forecast_method: str = "classic",
    forecast_length: int = 1000,
    internal_states: bool = False,
    feedback_metrics: bool = True,
        
    **kwargs,
):
    '''
    Forecast all files in a folder using a trained model.
    
    Args:
        trained_model_path (str): Path to the trained model
        data_folder (str): The folder containing the data files to be forecasted
        output_dir (str): Path for save the forecasted data

        forecast_method (str): The method to be used for forecasting. The default is ClassicForecast.
        forecast_length (int): The number of points to be forecasted. The default is 1000.

        section_initialization_length: int = 50,
        number_of_sections: int = 10,
    '''
    # get all files that end with .csv
    files = [_file for _file in os.listdir(data_folder) if _file.endswith('.csv')]

    print(f"Found {len(files)} files in {data_folder}")

    # Verify that the output directory exists, if not create it
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # forecast each file
    for i, _file in enumerate(files):

        print(f"Forecasting {_file}")
        print(f"File number {i+1} of {len(files)}")

        # check if file already exists and skip it if it does
        if os.path.isfile(os.path.join(output_dir, _file)):
            print(f"File {_file} already exists. Skipping...")
            continue

        # get the full path of the file
        data_file = os.path.join(data_folder, _file)

        # forecast the file
        forecast_from_saved_model(
            trained_model_path=trained_model_path,
            data_file=data_file,
            forecast_method=forecast_method,
            forecast_length=forecast_length,
            output_dir=output_dir,
            internal_states=internal_states,
            feedback_metrics=feedback_metrics,
            **kwargs,
        )
