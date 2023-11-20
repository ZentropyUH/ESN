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
from src.utils import load_data
from src.customs.custom_initializers import ErdosRenyi
from src.customs.custom_initializers import InputMatrix
from src.customs.custom_initializers import RegularNX
from src.customs.custom_initializers import WattsStrogatzNX
from src.plotters import plot_contourf_forecast
from src.plotters import plot_linear_forecast
from src.plotters import plot_rmse
from src.plotters import render_video


def train(
    data_file: str,
    model: str,
    units: int,
    steps: int,
    transient: int,
    train_length: int,
    input_initializer: str,
    input_bias_initializer: str,
    input_scaling: float,
    leak_rate: float,
    reservoir_activation: str,
    reservoir_initializer: str,
    reservoir_degree: int,
    reservoir_sigma: float,

    spectral_radius: float, #FIX ?
    regularization: float, #FIX
    readout_layer: str = None, #FIX

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
        steps
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

    ############### CHOOSE THE INPUT INITIALIZER ###############

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
            rewiring = kwargs["rewiring"]
            reservoir_initializer = WattsStrogatzNX(
                degree=reservoir_degree,
                spectral_radius=spectral_radius,
                rewiring_p=rewiring,
                sigma=reservoir_sigma,
            )
    
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
            overlap = kwargs["overlap"]
            reservoir_amount = kwargs["reservoir_amount"]
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

        case "Reservoir":
            raise Exception(f"{model} is yet to be implemented")
    
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
    steps: int = 1,
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
        step=steps,
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
            raise Exception(f"{forecast_method} is yet to be implemented")

        
    if output_dir:
        pd.DataFrame(predictions).to_csv(
            output_dir,
            index=False,
            header=None,
        )


    if internal_states:
        # Extraer el nombre base del archivo sin extensión
        file_name = os.path.splitext(os.path.basename(output_dir))[0]

         # Crear un directorio para los estados internos si no existe
        directory_path = os.path.dirname(output_dir)
        internal_state_dir = os.path.join(directory_path, "internal_state")
        os.makedirs(internal_state_dir, exist_ok=True)
        
        # Convertir los estados a lo largo del tiempo en un DataFrame de pandas y guardarlo en CSV
        states_over_time_df = pd.DataFrame(states_over_time)

        # Construir el nombre completo del archivo CSV para los estados internos
        internal_states_csv_path = os.path.join(internal_state_dir, f"{file_name}_states_over_time.csv")
        
        # Guardar los estados internos en la carpeta correspondiente
        states_over_time_df.to_csv(internal_states_csv_path, index=False, header=None)

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
):
    with open(join(trained_model_path, 'params.json')) as f:
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
        steps=params['steps'],
        internal_states=internal_states,
        feedback_metrics=feedback_metrics
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
