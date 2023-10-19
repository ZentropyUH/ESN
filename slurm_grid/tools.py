from os import listdir
from os import makedirs
from os.path import join
from os.path import isfile
from os.path import exists
import json
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict
from typing import List
from typing import Any
from itertools import product
from rich.progress import track

from slurm_grid.const import SLURM_SCRIPT
from slurm_grid.const import GridFolders
from slurm_grid.const import RunFolders
from slurm_grid.const import InfoFiles

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


def is_valid_file(filepath: str, extension: str = None):
    '''
    Raise an exeption if not valid file.
    
    Args:
        filepath (str): Path to the given file.

        extension (str, None): The extension of the file. If None, no extension will be checked.
    
    Return:
        None
    '''
    if not exists(filepath):
        raise FileNotFoundError
    if extension and not filepath.endswith(extension):
        raise Exception(f'{filepath} should be a {extension} file.')


# BUG: too many plots?
def save_plot(
    data: list,
    filepath: str,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
):
    '''
    Base method to plot data in a simple way.

    Args:
        data (list): Data to plot.

        filepath (str): Path to the output file to be generated.

        xlabel (str): Label for the x axis.

        ylabel (str): Label for the y axis.

        title (str): Title of the plot.
    
    Return:
        None
    '''
    plt.clf()
    plt.figure()
    plt.plot(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filepath)


def save_csv(data: np.ndarray, filepath: str):
    '''
    Save the data in a .csv file.

    Args:
        data (np.ndarray): Data to save.
        
        filepath (str): Path to the output file to be generated.
    
    Return:
        None
    '''
    pd.DataFrame(data).to_csv(
        filepath,
        index=False,
        header=None,
    )


def read_csv(filepath: str) -> np.ndarray:
    '''
    Read a .csv file.

    Args:
        filepath (str): Path to the .csv file to read.
    
    Return:
        (np.ndarray): The data fron the filepath in numpy format.
    '''
    is_valid_file(filepath, '.csv')
    return pd.read_csv(filepath).to_numpy()


def save_json(data: Dict, filepath: str):
    '''
    Save the data into a .json file with some options to make it more human readable.
    
    Args:
        data (Dict): A dictionary with the data to store in .json file.

        filepath (str): The path to the output file.
    '''
    with open(filepath, 'w') as f:
        json.dump(
            data,
            f,
            indent=4,
            sort_keys=True,
            separators=(",", ": "),
        )


def load_json(filepath: str) -> Dict:
    '''
    Load a .json file.

    Args:
        filepath (str): The path to the .json file to be loaded.
    
    Return:
        (Dict): Dictionary with the data from the .json file.
    '''
    is_valid_file(filepath, '.json')
    with open(filepath, 'r') as f:
        combinations = json.load(f)
        return combinations


def generate_combiantions(params: Dict[str, List[Any]]) -> Dict[str, Dict[str, Any]]:
    '''
    Generate all the combinations from the given params.

    Args:
        params (Dict): Dictionary with the params to generate the combinations.

    Return:
        (Dict): Dictionary with the combinations.
    '''
    param_name=[]
    param_value=[]
    for key in params.keys():
        param_name.append(key)
        param_value.append(params[key])
    
    data = {}
    for i, c in enumerate(product(*param_value)):
        data[i+1] = {pname: pvalue for pname, pvalue in zip(param_name, c)}
    return data


# FIX
# AUX
def _best_results(
    results_path: str,
    output: str,
    n_results: int,
    threshold: float
):
    '''
    # TODO
    Get the best results(the number of results by `n_results`) from the given `path` and save them in the `output` path.\n
    The results are the ones with the lowest mean square error.\n
    Will be stored in the `output` path with the folder name as the index of the result.\n
    Args:\n
        results_path (str): folder in 
        output (str):
        n_results (int):
        threshold (float):
        
    Return:\n
        None

    '''
    best = Queue(n_results)
    for folder in track(listdir(results_path), description='Searching best combinations'):
        folder = join(results_path, folder)
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
            params['leak_rate']
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
    # TODO
    Generate the new hyperparameters combinations from the results from `path` and save them in the `output` path.\n
    The new combinations are combinations of the old ones and +-10% of the steps of the old ones.\n
    Args:\n
    
    Return:\n
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
                params['leak_rate']
            ]
        )

    with open(steps_file, 'r') as f:
        steps_dict = json.load(f)
        steps_data = steps_dict['all']
        steps_data = [1 if index == 1 else i/10 for index, i in enumerate(steps_data) ]
        steps_dict['reservoir_sigma']=steps_data[0]
        steps_dict['reservoir_degree']=steps_data[1]
        steps_dict['regularization']=steps_data[2]
        steps_dict['spectral_radius']=steps_data[3]
        steps_dict['rewiring']=steps_data[4]
        steps_dict['leak_rate']=steps_data[5]
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
                if j == 2: #regularization
                    current_generated_combination.append([round(combinations[i][j]*steps_data[j], 10), combinations[i][j], round(combinations[i][j]/steps_data[j], 10)])
                elif j == 1 and combinations[i][j] <= 2: #reservoir degree
                    current_generated_combination.append([2, 3])
                elif j==4 or j==5: #rewiring or leak rate
                    d=round(combinations[i][j]+steps_data[j], 4)
                    m=combinations[i][j]
                    u= round(combinations[i][j]-steps_data[j], 4)
                    a=[]
                    if d<1 and d>0:
                        a.append(d)
                    if m<1 and m>0:
                        a.append(m)
                    if u<1 and u>0:
                        a.append(u)
                    current_generated_combination.append(a)

                else:
                    current_generated_combination.append([round(combinations[i][j]+steps_data[j], 4), combinations[i][j], round(combinations[i][j]-steps_data[j], 4)])
            new_combinations+=product(*current_generated_combination)

    new_combinations=set(new_combinations)    
    new_combinations = {i+1: x for i, x in enumerate(new_combinations)}

    with open(combinations_path, 'w') as f:
        json.dump(
            new_combinations,
            f,
            indent=4,
            sort_keys=True,
            separators=(",", ": "),
        )
    
    return new_combinations


def generate_slurm_script(
        job_name: str,
        array: tuple,
        combinations_path: str,
        output_path: str,
        data_path: str,
        filepath: str,
):
    '''
    # TODO
    Generate new slurm script.
    '''
    combinations_path:Path = Path(combinations_path)

    combinations_file = combinations_path.absolute().name
    output_path:Path = Path(output_path)
    data_path:Path = Path(data_path)

    file = SLURM_SCRIPT.format(
        job_name=job_name,
        array= "-".join([str(i) for i in array]),
        output_path=output_path,
        combinations_file=combinations_file,
        combinations_path=combinations_path,
        data_path=data_path.absolute(),
    )

    with open(filepath, 'w') as f:
        f.write(file)


def generate_unfinished_script(
        job_name: str,
        array: list,
        combinations_path: str,
        output_path: str,
        data_path: str,
        file_path:str
):
    '''
    # TODO
    Generate new slurm script for all the unfinished combinations.
    '''
    combinations_path: Path = Path(combinations_path)

    combinations_file = combinations_path.absolute().name
    output_path: Path = Path(output_path)
    data_path: Path = Path(data_path)

    file = SLURM_SCRIPT.format(
        job_name=job_name,
        array= ",".join([str(i) for i in array]),
        output_path=output_path,
        combinations_file=combinations_file,
        combinations_path=combinations_path,
        data_path=data_path.absolute(),
    )

    with open(file_path, 'w') as f:
        f.write(file)


def _results_data(
    results_path: str,
    filepath: str,
    threshold: float
):
    '''
    #TODO
    '''
    data = []
    for folder in track(listdir(results_path), description='Searching best combinations'):
        folder = join(results_path, folder)
        rmse_mean_path = join(folder, 'rmse_mean', 'rmse_mean.csv')
        params_path = join(folder, 'trained_model', 'params.json')

        with open(rmse_mean_path, 'r') as f:
            rmse_mean = read_csv(f)


        with open(params_path, 'r') as f:
            params = json.load(f)
        
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


def sort_by_int(array: List):
    '''
    Sort str list as int value.
    '''
    return [str(j) for j in sorted([int(i) for i in array])]


def _search_unfinished_combinations(
    path: str,
    depth: int,
    data_path: str,
):
    '''
    # TODO
    Search for the combinations that have not been satisfactorily completed and create a script to execute them
    path = specify the folder where the results of the combinations are stored
    depth = depth of the grid
    '''
    info_path = join(path, f'info_{depth}')
    comb_path = join(info_path, 'combinations.json')
    runs_path = join(path, f'run_{depth}', 'data')
    
    combinations = {}
    if exists(comb_path):
        with open(comb_path, 'r') as f:
            combinations = json.load(f)
    else:
        print(f'not exist {comb_path}')
        return

    unfinished = []
    folders = sort_by_int(listdir(runs_path))
    keys = sort_by_int(combinations.keys())
    for i in track(keys, description='Search unfinished runs'):
        if folders and i != folders[0]:
            unfinished.append(i)
        else:
            folders.pop(0)
            cpath = join(runs_path, i)
            if 'time.txt' not in listdir(cpath):
                unfinished.append(i)
                shutil.rmtree(cpath)

    if len(unfinished) == 0:
        print("All combinations terminated")
        return

    generate_unfinished_script(
        job_name="unfinished",
        array=unfinished,
        output_path=runs_path,
        data_path=data_path,
        combinations_path=comb_path,
        file_path=join(info_path, 'script_unfinished.sh')
    )


def _init_slurm_grid(
    path: str,
    job_name: str,
    data_path: str,

    model: str,
    input_initializer: str,
    input_bias_initializer: str,
    reservoir_activation: str,
    reservoir_initializer: str,

    units: List[int],
    train_length: List[int],
    forecast_length: List[int],
    transient: List[int],
    steps: List[int],

    input_scaling: List[float],
    leak_rate: List[float],
    spectral_radius: List[float],
    rewiring: List[float],
    reservoir_degree: List[int],
    reservoir_sigma: List[float],
    regularization: List[float],
):
    '''
    Main function to initialize the grid search.\n
    Generate the folders and the files needed to execute the grid search.
    '''
    run_path = join(path, GridFolders.RUN.value.format(depth=0))
    info_path = join(path, GridFolders.INFO.value.format(depth=0))
    output_path = join(run_path, RunFolders.RUN_DATA.value)
    params_path = join(info_path, InfoFiles.INFO_FILE.value)
    script_file = join(info_path, InfoFiles.SLURM_FILE.value)
    combinations_path = join(info_path, InfoFiles.COMBINATIONS_FILE.value)
    
    makedirs(info_path, exist_ok=True)
    makedirs(run_path, exist_ok=True)
    save_json(locals(), params_path)

    combinations = generate_combiantions(
        {
            'units':units,
            'train_length': train_length,
            'forecast_length': forecast_length,
            'transient': transient,
            'steps': steps,
            'input_scaling': input_scaling,
            'leak_rate': leak_rate,
            'spectral_radius': spectral_radius,
            'rewiring': rewiring,
            'reservoir_degree': reservoir_degree,
            'reservoir_sigma': reservoir_sigma,
            'regularization': regularization,
            'model': [model],
            'input_initializer': [input_initializer],
            'input_bias_initializer': [input_bias_initializer],
            'reservoir_activation': [reservoir_activation],
            'reservoir_initializer': [reservoir_initializer],
        }
    )

    save_json(combinations, combinations_path)
    generate_slurm_script(
        job_name,
        (1, len(combinations)),
        combinations_path,
        output_path,
        data_path,
        script_file,
    )


def _grid_aux(
    job_name: str,
    run_path: str,
    data_path: str,
    info_path: str,
    n_results: int,
    threshold: float,
    steps: int,
):
    '''
    #TODO
    '''
    output_path = join(run_path, 'data')
    results_path = join(run_path, 'results')
    steps_file = join(info_path, 'steps.json')
    new_info = join(Path(info_path).absolute().parent, str(int(Path(info_path).absolute().name)+1))
    makedirs(new_info, exist_ok=True)
    new_run = join(Path(run_path).absolute().parent, 'run_' + str(int(Path(run_path).absolute().name.split('_')[-1])+1))
    makedirs(new_run, exist_ok=True)

    _best_results(
        output_path,
        results_path,
        n_results,
        threshold
    )
    new_combinations = generate_result_combinations(
        results_path,
        steps_file,
        new_info,
    )
    generate_slurm_script(
        job_name,
        (1, len(new_combinations)),
        join(new_info, 'combinations.json'),
        join(new_run, 'data'),
        data_path,
        join(new_info, 'script.sh'),
        steps
    )