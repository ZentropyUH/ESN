import time
import json
from rich.progress import track

from functions import _train, _forecast
from src.grid.tools import *


'''
<output dir>
---- <name>
---- ---- trained_model
---- ---- forecast
---- ---- forecast_plots
---- ---- time.txt
---- ---- rmse
---- ---- rmse_mean
'''



def grid(
        data_path: str,
        output_path: str,

        units: int=6000,
        train_length: int=20000,
        forecast_length: int=1000,
        transient: int=1000,

        input_scaling=0.5,
        leak_rate=1.0,
        spectral_radius=0.99,
        rewiring=0.5,
        reservoir_degree=3,
        reservoir_sigma=0.5,
        regularization=1e-4,
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
    trained_model, train_params = _train(
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
            transient = train_params['transient'],
            train_length = train_params["train_length"],
            data_file= current_data,
            filepath= join(forecast_path, f'{fn}.csv'),
            forecast_length=forecast_length,
        )
        print('Forecasting {} finished'.format(fn))
        forecast_data.append((prediction, true_data))

        # PLOTS
        features = prediction.shape[-1]
        xvalues = np.arange(0, forecast_length)

        # Make each plot on a different axis
        fig, axs = plt.subplots(features, 1, sharey=True, figsize=(20, 9.6))
        fig.tight_layout()
        fig.subplots_adjust(top=0.9, bottom=0.08, hspace=0.3)

        fig.suptitle('', fontsize=16)
        fig.supxlabel('time')

        # Make this if-else better TODO
        if features == 1:
            axs.plot(xvalues, prediction[:, 0], label="prediction")
            axs.plot(xvalues, true_data[:, 0], label="target")
            axs.legend()
        else:
            for i in range(features):
                axs[i].plot(xvalues, prediction[:, i], label="prediction")
                axs[i].plot(xvalues, true_data[:, i], label="target")
                axs[i].legend()

        plt.savefig(join(forecast_plot_path, str(fn)))


    forecast_time = (time.time() - start_forecast_time)/len(data)
    
    # Calculate RMSE
    rmse = [np.sqrt(np.mean((pred - true_pred) ** 2, axis=1)) for pred, true_pred in forecast_data]

    # Sum all the rmse
    mean = []
    for i, current in enumerate(rmse):
        # Save current rmse
        save_csv(current, f'{i}.csv', rmse_path)
        
        if len(mean) == 0:
            mean = current
        else:
            mean = np.add(mean, current)

    mean = [x / len(data) for x in mean]

    # Save the csv
    save_csv(mean, "rmse_mean.csv", mean_path)
    save_plots(data=mean, output_path=mean_path, name='rmse_mean_plot.png')

    
    with open(time_file, 'w') as f:
        json.dump({'train': train_time, 'forecast': forecast_time}, f)



def _grid(
    units: int,
    train_length: int,
    forecast_length: int,
    transient: int,

    data_path: str,
    output_path: str,
    
    index: int,
    hyperparameters_path: str,
):
    
    params = load_hyperparams(hyperparameters_path)[str(index)]
    reservoir_sigma=params[0]
    reservoir_degree=params[1]
    regularization=params[2]
    spectral_radius=params[3]
    rewiring=params[4]
    input_scaling=0.5
    leak_rate=1.0

    grid(
        data_path=data_path,
        output_path=join(output_path, str(index)),
        units=units,
        train_length=train_length,
        forecast_length=forecast_length,
        transient=transient,

        input_scaling=input_scaling,
        leak_rate=leak_rate,
        spectral_radius=spectral_radius,
        rewiring=rewiring,
        reservoir_degree=reservoir_degree,
        reservoir_sigma=reservoir_sigma,
        regularization=regularization,
    )