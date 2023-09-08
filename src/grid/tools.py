from os import makedirs, listdir, system
from os.path import join, isdir, split, isfile
from pathlib import Path
from typing import Dict, List

from threading import Thread
from random import randint
from itertools import product, chain

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
# BUG: too many plots?
def save_plots(data: list, output_path: str, name: str):
    '''
    Save the plot of the given `data` in the given `output_path`/`name`.
    '''
    plt.clf()
    plt.figure()
    plt.plot(data)
    plt.xlabel("Time")
    plt.ylabel("Mean square error")
    plt.title("Plot of root mean square error")
    plt.savefig(join(output_path, name))



# Work with csv
def save_csv(data, name: str, path: str):
    '''
    Save the `data` in a .csv.
    '''
    pd.DataFrame(data).to_csv(
        join(path, name),
        index=False,
        header=None,
    )

def read_csv(file: str):
    '''
    Read a .csv file and return a numpy array.
    '''
    return pd.read_csv(file).to_numpy()



# Hyper Parameters
def load_hyperparams(filepath: str) -> Dict:
    '''
    Load the hyperparameters dictionary from a .json file.
    '''
    if not isfile(filepath) or not filepath.endswith('.json'):
        raise Exception(f'{filepath} is not a valid file')
    with open(filepath, 'r') as f:
        combinations = json.load(f)
        return combinations

def generate_combiantions(hyperparams: List[List[float]]):
    return list(product(*hyperparams))

# print(len(generate_combiantions([[0.2, 0.4, 0.6, 0.8, 1.0], [2, 4, 6, 8], [10e-5, 10e-6, 10e-7, 10e-8, 10e-9], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]])))

def generate_initial_combinations(output: str):
    '''
    Generate and save the initial hyperparameters combinations in the given path.\n
    The hyperparameters will be of the form: name: (initial_value, number_of_values, increment, function_of_increment).
    '''
    hyperparameters_to_adjust = {
        "sigma": (0.2, 5, 0.2, lambda x, y, i: round(x + y * i, 2)),
        "degree": (2, 4, 2, lambda x, y, i: round(x + y * i, 2)),
        "ritch_regularization": (10e-5, 5, 0.1, lambda x, y, i: round(x * y**i, 8)),
        "spectral_radio": (0.9, 16 , 0.01, lambda x, y, i: round(x + y * i, 2)),
        "reconection_prob": (0, 6, 0.2, lambda x, y, i: round(x + y*i, 2))
    }

    combinations = {
        int(i + 1): c
        for i, c in enumerate(
            product(
                *[[elem[3](elem[0], elem[2], i) for i in range(elem[1])] for elem in hyperparameters_to_adjust.values()]
            )
        )
    }

    with open(join(output, 'combinations.json'), 'w') as f:
        json.dump(
            combinations,
            f,
            indent=4,
            sort_keys=True,
            separators=(",", ": "),
        )
    
    steps = {
        "all": [
            0.2,
            2,
            0.1,
            0.01,
            0.2
        ],
        "sigma": 0.2,
        "degree": 2,
        "ritch_regularization": 0.1,
        "spectral_radio": 0.01,
        "reconection_prob": 0.2
    }
    with open(join(output, 'steps.json'), 'w') as f:
        json.dump(
            steps,
            f,
            indent=4,
            separators=(",", ": "),
        )
    return combinations



# AUX
def get_best_results(path: str, output: str, max_size: int, threshold: float):
    '''
    Get the best results(the number of results by `max_size`) from the given `path` and save them in the `output` path.\n
    The results are the ones with the lowest mean square error.\n
    Will be stored in the `output` path with the folder name as the index of the result.
    '''
    best = Queue(max_size)
    for folder in track(listdir(path), description='Searching best combinations'):
        folder = join(path, folder)
        rmse_mean_path = join(folder, 'rmse_mean', 'rmse_mean.csv')
        params_path = join(folder, 'trained_model', 'params.json')

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



def generate_result_combinations(
        path,
        steps_file,
        output,
):
    '''
    Generate the new hyperparameters combinations from the results from `path` and save them in the `output` path.\n
    The new combinations are combinations of the old ones and +-10% of the steps of the old ones.
    '''

    # PATHS
    steps_path = join(output, 'steps.json')
    combinations_path = join(output, 'combinations.json')
    makedirs(output, exist_ok=True)

    combinations = []
    steps_data = []

    for folder in listdir(path):
        folder = join(path, folder)
        params_path = join(folder, 'trained_model', 'params.json')

        with open(params_path, 'r') as f:
            params = json.load(f)

        combinations.append(
            [
                params['reservoir_sigma'],
                params['reservoir_degree'],
                params['regularization'],
                params['spectral_radius'],
                params['rewiring'],
            ]
        )

    with open(steps_file, 'r') as f:
        steps_dict = json.load(f)
        steps_data = steps_dict['all']
        steps_data = [1 if index == 1 else i/10 for index, i in enumerate(steps_data) ]

        steps_dict['all'] = steps_data
    
    with open(steps_path, 'w') as f:
        json.dump(
            steps_dict, 
            f,
            indent=4,
            separators=(",", ": ")
        )

    new_combinations = []

    for i in track(range(len(combinations)), description='Generating new combinations'):
        current_generated_combination = []
        for j in range(len(combinations[i])):
            if combinations[i][j] < 10e-15:
                current_generated_combination.append([round(combinations[i][j],4)])
            else:
                if j == 2:
                    current_generated_combination.append([round(combinations[i][j]*steps_data[j], 10), combinations[i][j], round(combinations[i][j]/steps_data[j], 10)])
                elif j == 1 and combinations[i][j] <= 2:
                    current_generated_combination.append([3, 2])
                else:
                    current_generated_combination.append([round(combinations[i][j]+steps_data[j], 4), combinations[i][j], round(combinations[i][j]-steps_data[j], 4)])
    
        new_combinations.append(product(*current_generated_combination))
    new_combinations = {i+1: x for i, x in enumerate(chain(*new_combinations))}

    with open(combinations_path, 'w') as f:
        json.dump(
            new_combinations,
            f,
            indent=4,
            sort_keys=True,
            separators=(",", ": "),
        )
    
    return new_combinations



def script_generator(
        job_name: str,
        array: tuple,
        combinations_path: str,
        output_path: str,
        data_path: str,
        filepath: str,
        steps:int
):
    '''
    Generate new slurm script.
    '''
    combinations_path:Path = Path(combinations_path)

    combinations_file = combinations_path.absolute().name
    output_path:Path = Path(output_path)
    data_path:Path = Path(data_path)

    file = f'''#!/bin/bash

########## RESOURCES TO USE ##########

#SBATCH --job-name="{job_name}"

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2000M

#SBATCH --gres=gpu:A100:1

#SBATCH --time=4-00:00:00
#SBATCH --partition=graphic

#SBATCH --array={array[0]}-{array[1]}


########## MODULES ##########

set -e
module purge
module load python/3.10.5

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK


########## PATHS ##########

# Create your scratch space
scratch="/scratch/$USER/$SLURM_JOB_ID"
mkdir -p $scratch
cd $scratch

# project path
ESN=./ESN
mkdir -p $ESN

# output path
output="./output"
mkdir -p $output

# data path
data="./data"
mkdir -p $data

# save path
save="{output_path}"
mkdir -p $save

# combinations
comb=$ESN/{combinations_file}


########## COPY ##########

# Copy project files to scratch
echo "copying project............"
cp -r /data/tsa/destevez/dennis/ESN/* $ESN

echo "copying project............"
cp -r "{combinations_path}" $ESN

echo "copying data............"
cp -r {data_path.absolute()}/* $data
echo "end of copy"



########## RUN ##########

echo "runing............"
srun python3 ESN/main.py grid -u 5000 -tl 20000 -fl 1000 -tr 1000 -d $data -o $output -i $SLURM_ARRAY_TASK_ID -hp $comb -s {steps}
echo "end of run"



########## SAVE ##########

echo "saving............"
cp -r $output/* $save
echo "end of save"



########## CLEANUP & EXIT ##########

# Clean up all the shit
rm -rf $scratch

# Exit gracefully
exit 0

########## END ##########
    '''

    with open(filepath, 'w') as f:
        f.write(file)



def results_info(path: str, filepath: str, threshold: float):
    data = []
    for folder in track(listdir(path), description='Searching best combinations'):
        folder = join(path, folder)
        rmse_mean_path = join(folder, 'rmse_mean', 'rmse_mean.csv')
        params_path = join(folder, 'trained_model', 'params.json')

        with open(rmse_mean_path, 'r') as f:
            rmse_mean = read_csv(f)


        with open(params_path, 'r') as f:
            params = json.load(f)
        
        params = {
            'reservoir_sigma': params['reservoir_sigma'],
            'reservoir_degree': params['reservoir_degree'],
            'regularization': params['regularization'],
            'spectral_radius': params['spectral_radius'],
            'rewiring': params['rewiring'],
        }
        
        index = len(rmse_mean)
        for i, x in enumerate(rmse_mean):
            if x > threshold:
                index = i
                break
        
        data.append({
            'index': index,
            'params': params,
            'folder': folder,
        })

    with open(filepath, 'w') as f:
        json.dump(
            data,
            f,
            indent=4,
            sort_keys=True,
            separators=(",", ": "),
        )

