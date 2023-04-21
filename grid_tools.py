from os import makedirs, listdir, system
from os.path import join, isdir

from threading import Thread
from random import randint
from itertools import product

import pandas as pd
import numpy as np

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
    
    
    def decide(self, l:list, combination, threshold):
        for i, x in enumerate(l):
            if x > threshold:
                self.add(i, combination)
                break
            elif i == len(l) - 1:
                self.add(i, combination)
                break



def save_csv(data, name:str, path:str):
    pd.DataFrame(data).to_csv(
    join(path, name),
    index=False,
    header=None,
    )


def generate_combinations(params:dict):
    return [[elem[3](elem[0], elem[2], i) for i in range(elem[1])] for elem in params.values()]



# main train
def train(params, data_file_path, output_file, u, tl, tn):    
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
def forecast(prediction_steps: int, train_transient: int, trained_model_path: str, prediction_path: str, data_file, forecast_name, trained: bool):    
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