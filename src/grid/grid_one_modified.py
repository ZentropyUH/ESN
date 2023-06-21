from src.grid.grid_tools import *
import time
import tensorflow as tf
import argparse
import json
from rich.progress import track
from model_functions import _forecast, _train, _plot
import os

from itertools import product


def generate_new_combinations(current_path, data_file='combination_threshold_time.json', count=5,  intervalse_len_file='intervals_len.json'):
    combinations = []
    len_intervals = []
    path = f"{current_path}{data_file}"
    path_len = f"{current_path}{intervalse_len_file}"
    with open(path, 'r') as f:
        data = json.load(f)
        data = sorted(data, key=lambda x: x[1])
        for i in range(min(count,len(data))):
            combinations.append(data[i][0])

    with open(path_len, 'r+') as f:
        int_len = json.load(f)
        len_intervals = int_len['all']
        len_intervals = [i/10 for i in len_intervals]
        int_len['all']=len_intervals
    with open(path_len, 'w') as f:
        json.dump(int_len, f)


    new_combinations = []
    for i in range(len(combinations[0])):
        temp = []
        for j in range(len(combinations)):
            if len(len_intervals) == i+2:
                temp.append(combinations[j][i]*len_intervals[i])
                temp.append(combinations[j][i]/len_intervals[i])
            else:
                temp.append(combinations[j][i]+len_intervals[i])
                temp.append(combinations[j][i]-len_intervals[i])

        new_combinations.append(temp)
    new_combinations = product(*new_combinations)
    new_combinations = {str(i): x for i, x in enumerate(new_combinations)}

    with open(join(current_path, 'combinations.json'), 'w') as data:
        json.dump(new_combinations, data)


def threshold_time(mean: list, threshold: float):
    threshold_time = 0
    for i in mean:
        if i > threshold:
            mean = i
            break
        threshold_time += 1
    return threshold_time


def save_combinations_threshold_time(combination, threshold_time, current_path=""):
    '''saves a list of list where first elem is combination of parameters and the second elem is the is the moment when the threshold is crossed'''
    data = []
    # combination = str(combination)
    # data[combination] = threshold_time
    data.append((combination, threshold_time))
    path = f"{current_path}combination_threshold_time.json"
    if not os.path.exists(path):
        with open(path, 'w') as f:
            json.dump(data, f)

    else:
        with open(path, 'r') as f:
            data = json.load(f)
            print(data)
            # data[combination] = threshold_time
            data.append((combination, threshold_time))
            print(data)
        with open(path, 'w') as f:
            json.dump(data, f)


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


def grid_one(combination_index: int, data_path: str, output_path: str, u: int = 9000, tl: int = 20000, combinations_path: str = 'src/grid/combinations.json', threshold=0.01):

    # Select the data to train
    data: list[str] = [join(data_path, p) for p in listdir(data_path)]
    train_index = randint(0, len(data) - 1)
    train_data_path = data[train_index]

    # Create the output folder
    makedirs(output_path, exist_ok=True)

    # Get the combination from src/grid/combinations.json file in the index position
    combination = []
    with open(combinations_path, 'r') as f:
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
    trained_model, trained_params = _train(
        data_file_path=train_data_path,
        output_file=current_path,
        file_name='trained_model',
        save_model=True
    )

    print('Training finished')
    train_time = time.time() - start_train_time

    # Forecast aqui se hace con el modelo no con el path
    start_forecast_time = time.time()
    for fn, current_data in enumerate(data):
        print('Forecasting {}...'.format(fn))
        _forecast(
            trained_model=trained_model,
            data_file=current_data,
            output_dir=forecast_path,
            model_params=trained_params
        )
        print('Forecasting {} finished'.format(fn))

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

    save_combinations_threshold_time(
        combination, threshold_time(threshold, mean), current_path)
    
    with open(time_file, 'w') as f:
        json.dump({'train': train_time, 'forecast': forecast_time}, f)
