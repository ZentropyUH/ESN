import time
import json
import numpy as np
from random import randint
from os.path import join
from os import makedirs, listdir

from functions import _train, _forecast
from slurm_grid.tools import (
    plot_prediction,
    load_hyperparams,
    save_plots,
    save_csv,
)



def grid(
        data_path: str,
        output_path: str,

        units: int,
        train_length: int,
        forecast_length: int,
        transient: int,
        steps: int,

        input_scaling: float,
        leak_rate: float,
        spectral_radius: float,
        rewiring: float,
        reservoir_degree: int,
        reservoir_sigma: float,
        regularization: float,
    ):
    # Select the data to train
    data: list[str] = [join(data_path, p) for p in listdir(data_path)]
    train_index = randint(0, len(data) - 1)
    train_data_path = data[train_index]

    current_path = output_path
    makedirs(current_path, exist_ok=True)
    
    # Create forecast folder
    forecast_path = join(current_path, 'forecast')
    makedirs(forecast_path, exist_ok=True)
    
    # Create folder for the rmse of predictions
    rmse_path = join(current_path, 'rmse')
    makedirs(rmse_path, exist_ok=True)

    # Create the folder mean
    mean_path = join(current_path, 'rmse_mean')
    makedirs(mean_path, exist_ok=True)

    # Create Trained model file
    trained_model_path = join(current_path, 'trained_model')
    makedirs(trained_model_path, exist_ok=True)

    forecast_plot_path = join(current_path, 'forecast_plots')
    makedirs(forecast_plot_path)

    # Create time file
    time_file = join(current_path, 'time.txt')


    # Train
    start_train_time = time.time()
    print('Training...')

    # Se manda a entrenar con los parametros por defecto, en este caso
    trained_model = _train(
        data_file=train_data_path,
        filepath=trained_model_path,
        
        model='ESN',
        input_initializer='InputMatrix',
        input_bias_initializer='RandomUniform',
        reservoir_activation='tanh',
        reservoir_initializer='WattsStrogatzNX',

        # seed=42,
        units=units,
        transient=transient,
        train_length=train_length,
        steps=steps,

        input_scaling=input_scaling,
        leak_rate=leak_rate,
        spectral_radius=spectral_radius,
        rewiring=rewiring,
        reservoir_degree=reservoir_degree,
        reservoir_sigma=reservoir_sigma,
        regularization=regularization,
    )

    print('Training finished')
    train_time = time.time() - start_train_time

    # Forecast aqui se hace con el modelo no con el path
    start_forecast_time = time.time()
    forecast_data = []
    for fn, current_data in enumerate(data):
        print('Forecasting {}...'.format(fn))
        prediction, true_data = _forecast (
            trained_model = trained_model,
            transient = transient,
            train_length = train_length,
            data_file= current_data,
            filepath= join(forecast_path, f'{fn}.csv'),
            forecast_length=forecast_length,
            steps=steps,
        )
        print('Forecasting {} finished'.format(fn))
        forecast_data.append((prediction, true_data))

        # PLOTS
        plot_prediction(
            data=true_data,
            prediction=prediction,
            filepath=join(forecast_plot_path, str(fn)),
            dt=1,
        )

    forecast_time = (time.time() - start_forecast_time)/len(data)
    
    # Calculate RMSE
    rmse = [np.sqrt(np.mean((pred - true_pred) ** 2, axis=1)) for pred, true_pred in forecast_data]

    # Sum all the rmse
    mean = []
    for i, current in enumerate(rmse):
        # Save current rmse
        save_csv(current, join(rmse_path, f'{i}.csv'))
        
        if len(mean) == 0:
            mean = current
        else:
            mean = np.add(mean, current)

    mean = [x / len(data) for x in mean]

    # Save the csv
    save_csv(mean, join(mean_path, 'rmse_mean.csv'))
    save_plots(data=mean, output_path=mean_path, name='rmse_mean_plot.png')

    
    with open(time_file, 'w') as f:
        json.dump({'train': train_time, 'forecast': forecast_time}, f)



def _slurm_grid(
    data_path: str,
    output_path: str,
    index: int,
    hyperparameters_path: str,
):  
    params = load_hyperparams(hyperparameters_path)[str(index)]

    grid(
        data_path=data_path,
        output_path=join(output_path, str(index)),
        units=params['units'],
        train_length=params['train_length'],
        forecast_length=params['forecast_length'],
        transient=params['transient'],
        steps=params['steps'],

        input_scaling=params['input_scaling'],
        leak_rate=params['leak_rate'],
        spectral_radius=params['spectral_radius'],
        rewiring=params['rewiring'],
        reservoir_degree=params['reservoir_degree'],
        reservoir_sigma=params['reservoir_sigma'],
        regularization=params['regularization'],
    )