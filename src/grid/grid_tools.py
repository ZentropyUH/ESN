from os import makedirs, listdir, system
from os.path import join, isdir, split, isfile

from threading import Thread
from random import randint
from itertools import product

import pandas as pd
import numpy as np
import csv
import json
import shutil
from rich.progress import track
import matplotlib.pyplot as plt



class Queue:
    '''
    Priority Queue with limited size, sorted from max to min.
    '''
    def __init__(self, max_size: int):
        self.queue = []
        self.max_size = max_size

    def add(self, val, data):
        if not self.queue:
            self.queue.append((val, data))
        else:
            for i, v in enumerate(self.queue):
                if val >= v[0]:
                    self.queue.insert(i, (val, data))
                    break

        if len(self.queue) > self.max_size:
            self.queue.pop()

    def decide(self, l: list, combination: tuple, folder: str, threshold: int):
        for i, x in enumerate(l):
            if x > threshold:
                self.add(i, (combination, folder))
                break

# TODO: make base method for plots
def save_plots(data: list, output_path: str, name: str):
    plt.clf()
    plt.figure()
    plt.plot(data)
    plt.xlabel("Time")
    plt.ylabel("Mean square error")
    plt.title("Plot of root mean square error")
    plt.savefig(join(output_path, name))

def save_plots_from_csv(input_path, output_path, name):
    with open(input_path, "r") as archivo:
        data = list(csv.reader(archivo))

    data = [float(date[0]) for date in data]
    plt.clf()
    plt.figure()
    plt.plot(data)
    plt.xlabel("Time")
    plt.ylabel("Mean square error")
    plt.title("Plot of root mean square error")
    plt.savefig(output_path + name)

# DELETE?
def save_plots_data_forecast(data: str, forecast: str, output: str):
    plt.clf()
    plt.figure()
    plt.plot(data, color="blue")
    plt.plot(forecast, color="red")
    plt.xlabel("Time")
    plt.ylabel("value")
    plt.title("Plot data and forecast")
    plt.savefig(output)

# DELETE?
def get_axis(data, axis: int):
    return [x[axis] for x in data]

# DELETE?
def plots_data_forecast(data: str, forecast: str, output: str):
    save_plots_data_forecast(
        get_axis(data, 0), get_axis(forecast, 0), output + "_x"
    )
    save_plots_data_forecast(
        get_axis(data, 1), get_axis(forecast, 1), output + "_y"
    )
    save_plots_data_forecast(
        get_axis(data, 2), get_axis(forecast, 2), output + "_z"
    )


# Work with csv
def save_csv(data, name: str, path: str):
    pd.DataFrame(data).to_csv(
        join(path, name),
        index=False,
        header=None,
    )

def read_csv(file: str):
    return pd.read_csv(file).to_numpy()


# Hyper Parameters
def load_hyperparams(filepath: str):
    if not isfile(filepath) or not filepath.endswith('.json'):
        raise Exception(f'{filepath} is not a valid file')
    with open(filepath, 'r') as f:
        combinations = json.load(f)
        return combinations

def generate_combinations(params: dict):
    return [
        [elem[3](elem[0], elem[2], i) for i in range(elem[1])]
        for elem in params.values()
    ]

def get_param_tuple(value, param, step):
    initial_value, number_of_values, _, function_of_increment = param
    initial_value = value - int(number_of_values / 2) * step
    return initial_value, number_of_values, step, function_of_increment

def get_ritch_param_tuple(value, param, step):
    initial_value, number_of_values, _, function_of_increment = param
    initial_value = value / int(number_of_values / 2) * step
    return initial_value, number_of_values, step, function_of_increment

# DELETE?
def save_combinations_txt(hyperparameters_to_adjust: dict, path: str):
    with open(join(path, "combinations.txt"), "w") as f:
        for i, c in enumerate(
            product(*generate_combinations(hyperparameters_to_adjust))
        ):
            f.write(
                "{} {} {} {} {} {}\n".format(
                    i + 1, c[0], c[1], c[2], c[3], c[4]
                )
            )

def save_combinations(hyperparameters_to_adjust: dict):
    with open("./combinations.json", "w") as f:
        json.dump(
            {
                int(i + 1): c
                for i, c in enumerate(
                    product(*generate_combinations(hyperparameters_to_adjust))
                )
            },
            f,
            indent=4,
            sort_keys=True,
            separators=(",", ": "),
        )

def generate_combinations(filepath: str):
    # The hyperparameters will be of the form: name: (initial_value, number_of_values, increment, function_of_increment)
    # The parameters of the increment function are: initial_value, increment, current_value_of_the_iteration
    hyperparameters_to_adjust = {
        "sigma": (0.2, 5, 0.2, lambda x, y, i: round(x + y * i, 2)),
        "degree": (2, 4, 2, lambda x, y, i: round(x + y * i, 2)),
        "ritch_regularization": (10e-5, 5, 0.1, lambda x, y, i: round(x * y**i, 8)),
        "spectral_radio": (0.9, 16 , 0.01, lambda x, y, i: round(x + y * i, 2)),
        "reconection_prob": (0, 6, 0.2, lambda x, y, i: round(x + y*i, 2))
    }
    save_combinations(hyperparameters_to_adjust)




# AUX
def detect_not_fished_jobs(path: str, output: str):
    with open(join(output, "out.out"), "w") as f:
        for file in [
            join(path, f)
            for f in listdir(path)
            if "time.txt" not in listdir(join(path, f))
        ]:
            f.write("{}\n".format(file.split("_")[-1]))

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

# DELETE?
def calculate_aprox_time(time: list, file: str, text):
    with open(file, "a+") as f:
        f.write("{}: {}\n".format(text, str(np.mean(time))))