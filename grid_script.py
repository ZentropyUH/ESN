from os import makedirs, listdir, system
from os.path import join, isdir

from threading import Thread
from random import randint
from itertools import product

import pandas as pd
import numpy as np


'''
<output dir>
---- output
---- ---- <name>
---- ---- ---- trained_model
---- ---- ---- predictions
---- ---- ---- mse_mean
'''



def grid(hyperparameters_to_adjust:dict, data_path, output_path, u=5000, tl=1000, threshold=0.1):

    # List all the files on the data folder
    data: list[str] = [join(data_path, p) for p in listdir(data_path)]

    # Create the output folder
    output_path = join(output_path, 'output')
    makedirs(output_path, exist_ok=True)

    
    #Create a list[list] with the values of every hyperparameter
    params: list[list] = [[elem[3](elem[0], elem[2], i) for i in range(elem[1])] for elem in hyperparameters_to_adjust.values()]
    
    #Create all the combinations of hyperparameters
    for combination in product(*params):
        
        # Select the data to train
        train_index = randint(0, len(data) - 1)
        train_data_path = data[train_index]

        # Create the output folders
        current_path = join(output_path, '_'.join([str(x) for x in combination]))
        trained_model_path = join(current_path, 'trained_model')
        forecast_path = join(current_path, 'forecast')
        makedirs(forecast_path, exist_ok=True)

        # Train
        train(combination, train_data_path, current_path, u, tl, 'trained_model')

        # List of Threads
        forecast_list: list[Thread] = []
        
        for fn, current_data in enumerate(data):
            
            # Thread for forecast
            current = Thread(
                target = forecast,
                kwargs={
                    "prediction_steps": 1000,
                    "train_transient": tl,
                    "trained_model_path": trained_model_path,
                    "prediction_path": forecast_path,
                    "data_file": current_data,
                    "forecast_name": fn,
                    "trained": current_data == train_data_path,
                }
            )
            
            # Add Thread to queue
            forecast_list.append(current)
            # Start Thread
            current.start()
        
        # Wait for all Threads to finish
        for thread in forecast_list:
            thread.join()
        


        forecast_data = [join(forecast_path, x) for x in listdir(forecast_path)]


        mse = [[np.square(np.subtract(f, d)).mean()
                # for f1, d1 in zip(f, d)]
                for f, d in zip(pd.read_csv(forecast_file).to_numpy(), pd.read_csv(data_file).to_numpy())]
                for forecast_file, data_file in zip(forecast_data, data)]

        # Sum all the mse
        mean = []
        for current in mse:
            if mean == []:
                mean = current
            else:
                mean = np.add(mean, current)
                # list(map(lambda x, y: x+y, mean, current))

        mean = [x / len(data) for x in mean]
        
        # Create the folder mean
        mean_path = join(current_path, 'mse_mean')
        makedirs(mean_path, exist_ok=True)

        # Save the csv
        pd.DataFrame(mean).to_csv(
        join(mean_path, "mse_mean.csv"),
        index=False,
        header=None,
        )
        



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


# The hyperparameters will be of the form: name: (initial_value, number_of_values, increment, function_of_increment)
# The parameters of the increment function are: initial_value, increment, current_value_of_the_iteration
hyperparameters_to_adjust = {"sigma": (0.2, 5, 0.2, lambda x, y, i: round(x + y * i, 2)),
                        "degree": (2, 4, 2, lambda x, y, i: round(x + y * i, 2)),
                        "ritch_regularization": (10e-5, 5, 0.1, lambda x, y, i: round(x * y**i, 8)),
                        "spectral_radio": (0.9, 16 , 0.01, lambda x, y, i: round(x + y * i, 2)),
                        "reconection_prob": (0, 6, 0.2, lambda x, y, i: round(x + y*i, 2))
                    }


grid(hyperparameters_to_adjust, 
        data_path = '/media/dionisio35/Windows/_folders/_new/22/',         
        output_path = '/media/dionisio35/Windows/_folders/_new/') 
