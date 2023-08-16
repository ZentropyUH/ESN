from os import makedirs, listdir, system
from os.path import join, isdir, split, isfile
from pathlib import Path

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
def save_plots(data: list, output_path: str, name: str):
    plt.clf()
    plt.figure()
    plt.plot(data)
    plt.xlabel("Time")
    plt.ylabel("Mean square error")
    plt.title("Plot of root mean square error")
    plt.savefig(join(output_path, name))



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

def save_combinations(hyperparameters_to_adjust: dict):
    with open("./combinations.json", "w") as f:
        json.dump(
            {
                int(i + 1): c
                for i, c in enumerate(
                    product(
                        [[elem[3](elem[0], elem[2], i) for i in range(elem[1])] for elem in hyperparameters_to_adjust.values()]
                    )
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



def generate_new_combinations(
        path,
        steps_file,
        output,
):
    '''
    using the best combinations of the last run and the intervals len saved.
    '''

    # PATHS
    steps_path = join(output, 'steps.json')
    combinations_path = join(output, 'combinations.json')
    makedirs(output, exist_ok=True)

    combinations = []
    steps_data = []

    for folder in track(listdir(path), description='Generating new combinations'):
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
        steps_data = [((i-1 if i>0 else 0) if (index == 1) else i/10) for index,i in enumerate(steps_data) ]

        steps_dict['all'] = steps_data
    
    with open(steps_path, 'w') as f:
        json.dump(
            steps_dict, 
            f,
            indent=4,
            separators=(",", ": ")
        )

    new_combinations = []

    for i in range(len(combinations)):       
        new_combinations.append([]) 
        for j in range(len(combinations[i])):
            if len(steps_data)-3 == j:
                new_combinations[i].append(
                    [round(combinations[i][j]*steps_data[i], 10), round(combinations[i][j]/steps_data[i], 10)]
                )
            else:
                if combinations[i][j] < 10e-15:
                    new_combinations[i].append(
                        [round(combinations[i][j],4)]
                    )
                else:
                    new_combinations[i].append(
                    [round(combinations[i][j]+steps_data[i],4), round(combinations[i][j]-steps_data[i],4)]
                )
        
        new_combinations[i] = product(*new_combinations[i])
            

    new_combinations= chain(*new_combinations)
    new_combinations = {i+1: x for i, x in enumerate(k for k in new_combinations)}

    with open(combinations_path, 'w') as f:
        json.dump(
            new_combinations,
            f,
            indent=4,
            sort_keys=True,
            separators=(",", ": "),
        )
    
    return new_combinations



def script_generator(job_name: str, array: tuple, combinations: str, output: str, data: str, filepath: str):
    combinations:Path = Path(combinations)
    output:Path = Path(output)
    data:Path = Path(data)

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
save="{output}"
mkdir -p $save

# combinations
comb=$ESN/combinations.json


########## COPY ##########

# Copy project files to scratch
echo "copying project............"
cp -r /data/tsa/destevez/dennis/ESN/* $ESN

echo "copying project............"
cp -r "{combinations}" $ESN

echo "copying data............"
cp -r {data.absolute()}/* $data
echo "end of copy"



########## RUN ##########

echo "runing............"
srun python3 ESN/main.py grid -u 6000 -tl 20000 -fl 1000 -tr 1000 -d $data -o $output -i $SLURM_ARRAY_TASK_ID -hp $comb
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