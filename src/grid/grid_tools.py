from os import makedirs, listdir, system
from os.path import join, isdir, split

from threading import Thread
from random import randint
from itertools import product

import pandas as pd
import numpy as np
import csv
import json
import matplotlib.pyplot as plt

from src.functions import training, forecasting


# Priority Queue with limited size, sorted from max to min
class Queue:
    def __init__(self, max_size:int):
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
    
    
    def decide(self, l:list, combination: tuple, folder: str, threshold: int):
        for i, x in enumerate(l):
            if x > threshold:
                self.add(i, (combination, folder))
                break
            elif i == len(l) - 1:
                self.add(i, (combination, folder))
                break




def save_plots(data, output_path,name):
    plt.plot(data)
    plt.xlabel('Time')
    plt.ylabel('Mean square error')
    plt.title('Plot of root mean square error')
    plt.savefig(output_path + name)

def save_plots_from_csv(input_path, output_path, name):
    with open(input_path, 'r') as archivo:
        data = list(csv.reader(archivo))

    data = [float(date[0]) for date in data]
    plt.plot(data)
    plt.xlabel('Time')
    plt.ylabel('Mean square error')
    plt.title('Plot of root mean square error')
    plt.savefig(output_path + name)


def save_csv(data, name:str, path:str):
    pd.DataFrame(data).to_csv(
    join(path, name),
    index=False,
    header=None,
    )


def read_csv(file: str):
    return pd.read_csv(file).to_numpy()


def generate_combinations(params:dict):
    return [[elem[3](elem[0], elem[2], i) for i in range(elem[1])] for elem in params.values()]

def get_param_tuple(value, param , step):    
    initial_value, number_of_values, _ , function_of_increment = param
    initial_value = value - int(number_of_values / 2) * step
    return initial_value, number_of_values, step, function_of_increment

def get_ritch_param_tuple(value, param , step):    
    initial_value, number_of_values, _ , function_of_increment = param
    initial_value = value / int(number_of_values / 2) * step
    return initial_value, number_of_values, step, function_of_increment

def calculate_aprox_time(time: list, file: str, text):
    with open(file, 'a+') as f:
        f.write('{}: {}\n'.format(text, str(np.mean(time))))


def save_combinations_txt(hyperparameters_to_adjust: dict, path: str):
    with open(join(path, "combinations.txt"), "w") as f:
        for i, c in enumerate(product(*generate_combinations(hyperparameters_to_adjust))):
            f.write("{} {} {} {} {} {}\n".format(i+1, c[0], c[1], c[2], c[3], c[4]))

def save_combinations(hyperparameters_to_adjust: dict):
    with open("./combinations.json", "w") as f:
        json.dump({int(i+1): c for i, c in enumerate(product(*generate_combinations(hyperparameters_to_adjust)))}, f, indent=4, sort_keys=True, separators=(',', ': '))


# main train
def train_main(params, data_file_path, output_file, u, tl, tn):    
    instruction = f"python3 ./main.py train \
            -m ESN \
            -ri WattsStrogatzOwn\
            -df {data_file_path} \
            -o {output_file} \
            -rs {params[0]} \
            -sr {params[3]} \
            -rw {params[4]} \
            -u {u} \
            -tl {tl} \
            -rd {params[1]} \
            -tn {tn} \
            -rg {params[2]}"

    system(instruction)


# main forecast
def forecast_main(prediction_steps: int, train_transient: int, trained_model_path: str, prediction_path: str, data_file, forecast_name, trained: bool):    
    if trained:
        instruction = f"python3 ./main.py forecast \
                -fm classic \
                -fl {prediction_steps} \
                -it {1000} \
                -tr {train_transient} \
                -tl {train_transient} \
                -tm {trained_model_path} \
                -df {data_file} \
                -o {prediction_path} \
                -fn {forecast_name}"
    else:
        instruction = f"python3 ./main.py forecast \
                -fm classic \
                -fl {prediction_steps} \
                -it {1000} \
                -tr {0} \
                -tl {train_transient}\
                -tm {trained_model_path} \
                -df {data_file} \
                -o {prediction_path} \
                -fn {forecast_name}"

    system(instruction)



def train(params, data_file_path, output_file, u, tl, tn):
    training(
        model='ESN',
        units=str(u),
        input_initializer='InputMatrix',
        input_scaling='0.5',
        input_bias_initializer='RandomUniform',
        leak_rate='1.',
        reservoir_activation='tanh',
        spectral_radius=str(params[3]),
        reservoir_initializer='WattsStrogatzOwn',
        rewiring=str(params[4]),
        reservoir_degree=str(params[1]),
        reservoir_sigma=str(params[0]),
        reservoir_amount=None,
        overlap=None,
        readout_layer='linear',
        regularization=str(params[2]),
        init_transient=1000,
        transient=1000,
        train_length=str(tl),
        output_dir=output_file,
        data_file=data_file_path,
        trained_name=tn
    )

def forecast(prediction_steps: int, train_transient: int, trained_model_path: str, prediction_path: str, data_file, forecast_name, trained: bool):
    forecasting(
        forecast_method='classic',
        forecast_length=prediction_steps,
        section_initialization_length=50,
        number_of_sections=10,
        init_transient=1000,
        transient=1000 if trained else train_transient,
        train_length=train_transient,
        trained_model=trained_model_path,
        output_dir=prediction_path,
        data_file=data_file,
        forecast_name=str(forecast_name)
    )