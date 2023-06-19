from src.grid.grid_tools import *
import time
import tensorflow as tf
import argparse
import json
from rich.progress import track
from model_functions import _train, _forecast


def calculate_mse(forecast_path: str, data_path: str, output, t: int):
    forecast_data = [join(forecast_path, x) for x in listdir(forecast_path)]
    data: list[str] = [join(data_path, p) for p in listdir(data_path)]
    makedirs(output, exist_ok=True)
        
    # Calculate RMSE
    rmse = [[np.sqrt(((f - d) ** 2).mean())
            for f, d in zip(pd.read_csv(forecast_file).to_numpy(), pd.read_csv(data_file).to_numpy()[(t):])]
            for forecast_file, data_file in zip(forecast_data, data)]

    # Sum all the rmse
    mean = []
    for i, current in enumerate(rmse):
        
        # Save current rmse
        save_csv(current, "{}.csv".format(i), output)
        
        if len(mean) == 0:
            mean = current
        else:
            mean = np.add(mean, current)

    mean = [x / len(data) for x in mean]

    # Save the csv
    save_csv(mean, "rmse_mean.csv", output)
    save_plots(data=mean, output_path=output, name='rmse_mean_plot.png')



def grid_one(combination_index: int, data_path: str, output_path:str, u:int=9000, tl:int=20000):

        # Select the data to train
        data: list[str] = [join(data_path, p) for p in listdir(data_path)]
        train_index = randint(0, len(data) - 1)
        train_data_path = data[train_index]

        # Create the output folder
        makedirs(output_path, exist_ok=True)

        # Get the combination from src/grid/combinations.json file in the index position
        combination = []
        with open('src/grid/combinations.json', 'r') as f:
            combinations = json.load(f)
            combination = combinations[str(combination_index)]
        print('Combination: {}'.format(combination))
        

        # Create the trained model folder
        current_path = join(output_path, '_'.join([str(x) for x in combination]))
        makedirs(current_path)
        
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
        (trained_model, train_params) = _train (

            data_file_path=train_data_path,
            output_file=current_path,
            file_name ='trained_model',
            
        )

        print('Training finished')
        train_time = time.time() - start_train_time

        # Forecast aqui se hace con el modelo no con el path
        start_forecast_time = time.time()
        for fn, current_data in enumerate(data):
            print('Forecasting {}...'.format(fn))
            _forecast(
                trained_model = trained_model,
                model_params = train_params
                data_file= current_data,
                output_dir= forecast_path,
            )
            print('Forecasting {} finished'.format(fn))

            plot_main(
                prediction_file=join(forecast_path, str(fn)),
                data_file=current_data,
                tl=tl,
                output_file=join(forecast_plot_path, str(fn))
            )

        forecast_time = (time.time() - start_forecast_time)/len(data)

        # Get Forecast data files
        forecast_data = [join(forecast_path, x) for x in listdir(forecast_path)]
        
        # Calculate RMSE
        rmse = [[np.sqrt(((f - d) ** 2).mean())
                for f, d in zip(pd.read_csv(forecast_file).to_numpy(), pd.read_csv(data_file).to_numpy()[(1000 + 1000 + tl):])]
                for forecast_file, data_file in zip(forecast_data, data)]

        # Sum all the rmse
        mean = []
        for i, current in enumerate(rmse):
            
            # Save current rmse
            save_csv(current, "{}.csv".format(i), rmse_path)
            
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


def best_combinations(path: str, output: str, max_size: int, threshold: float):
    
    best = Queue(max_size)
    for folder in track(listdir(path), description='Searching best combinations'):
        folder = join(path, folder)
        rmse_mean_path = join(folder, 'rmse_mean', 'rmse_mean.csv')
        params_path = join(folder, 'params.json')

        rmse_mean = []
        with open(rmse_mean_path, 'r') as f:
            rmse_mean = read_csv(f)
        
        params = {}
        with open(params_path, 'r') as f:
            params = json.load(f)
        
        params = (
            params['reservoir_sigma'],
            params['reservoir_degree'],
            params['regularization'],
            params['spectral_radius'],
            params['rewiring'],
        )
        
        best.decide(rmse_mean, params, folder, threshold)
    
    for i, element in enumerate(best.queue):
        folder = element[1][1]
        shutil.copytree(folder, join(output, str(i)), dirs_exist_ok=True)

