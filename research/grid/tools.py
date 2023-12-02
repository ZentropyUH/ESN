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

from research.grid.const import SLURM_SCRIPT
from research.grid.const import GridFolders
from research.grid.const import RunFolders
from research.grid.const import InfoFiles
from research.grid.const import CaseRun


class Queue:
    '''
    Priority Queue with limited size, sorted from max to min.
    '''
    def __init__(self, max_size: int):
        self.queue = []
        self.max_size = max_size

    def add(self, val: int, data: Any, greater: bool = True):
        '''
        Add a new element to the queue. If the queue is full, the element with the highest value will be removed.

        Args:
            val (int): Value to compare in queue.

            data (Any): data to store in queue.
        
        Return:
            None
        '''
        if not self.queue:
            self.queue.append((val, data))
        else:
            for i, v in enumerate(self.queue):
                if greater:
                    if val >= v[0]:
                        self.queue.insert(i, (val, data))
                        break
                else:
                    if val <= v[0]:
                        self.queue.insert(i, (val, data))
                        break

        if len(self.queue) > self.max_size:
            self.queue.pop()


def find_index(elements: List[float], threshold: int):
    '''
    Find the index where the threshold is exceeded.

    Args:
        elements (List[float]): List of values to use to find the index that exceed the threshold.

        data (Any): Data to store in queue.

        threshold (float): The threshold to decide if the result is good or not.
    
    Return:
        None
    '''
    for i, x in enumerate(elements):
        if x > threshold:
            return i
            break
    return len(elements)


def is_valid_file(filepath: str, extension: str = None):
    '''
    Raise an exeption if not valid file.
    
    Args:
        filepath (str): Path to the given file.

        extension (str, None): The extension of the file. If None, no extension will be checked.
    
    Return:
        None
    '''
    if not isfile(Path(filepath)):
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


# AUX
def best_results(
    results_path: str,
    output: str,
    n_results: int,
):
    '''
    Get the best results fron the giver path.(the number of results by `n_results`) from the given `path` and save them in the `output` path.\n
    The results are the ones with the lowest index that by which it exceeds the threshold fron the mean square error.\n

    Args:
        results_path (str): Path to the data to work with.

        output (str): Path to save the output. Will be stored with the folder name as the index of the result.

        n_results (int): The number of best results to save.

        threshold (float): The threshold to decide if the result is good or not.
    
        Return:
            None
    '''
    best = Queue(n_results)
    for folder in track(listdir(results_path), description='Searching best combinations'):
        folder = join(results_path, folder)
        eval_file = join(folder, CaseRun.EVALUATION_FILE.value)
        eval_json = load_json(eval_file)
        
        nrmse = eval_json['nrmse']
        best.add(nrmse, folder, greater=False)
    
    for i, element in enumerate(best.queue):
        folder = element[1]
        shutil.copytree(folder, join(output, str(i)), dirs_exist_ok=True)


def generate_result_combinations(
        path,
        steps_file,
        output,
):
    '''
    Generate the new hyperparameters combinations from the results from `path` and save them in the `output` path.\n
    The new combinations are combinations of the old ones and +-10% of the steps of the old ones.\n
    Args:\n
        path (str): folder directory where the results are located 
        steps_file (str): directory of the .json file where the step sizes are located 
        output (str): folder directory where the new combinations will be saved
    Return:\n
        new_combinations(dict): all the new combinations
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
        jobs_limit: int,
):
    '''
    Generate new slurm script.

    Args:
        job_name (str): Name of the job.

        array (tuple): Tuple with the range of the array slurm job.

        combinations_path (str): Path to the combinations file.

        output_path (str): Path to the output folder.

        data_path (str): Path to the data folder.

        filepath (str): Path to the output file to be generated as .sh.
    
    Return:
        None
    '''
    combinations_path:Path = Path(combinations_path)

    combinations_file = combinations_path.absolute().name
    output_path:Path = Path(output_path)
    data_path:Path = Path(data_path)

    file = SLURM_SCRIPT.format(
        repo=Path(__file__).parent.parent.parent.absolute(),
        job_name=job_name,
        array= "-".join([str(i) for i in array]),
        output_path=output_path,
        combinations_file=combinations_file,
        combinations_path=combinations_path,
        data_path=data_path.absolute(),
        jobs_limit=jobs_limit,
    )

    with open(filepath, 'w') as f:
        f.write(file)


def generate_unfinished_script(
        job_name: str,
        array: list,
        combinations_path: str,
        output_path: str,
        data_path: str,
        file_path:str,
        jobs_limit: int,
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
        repo=Path(__file__).parent.parent.parent.absolute(),
        job_name=job_name,
        array= array,
        output_path=output_path,
        combinations_file=combinations_file,
        combinations_path=combinations_path,
        data_path=data_path.absolute(),
        jobs_limit=jobs_limit,
    )

    with open(file_path, 'w') as f:
        f.write(file)


def results_data(
    results_path: str,
    filepath: str,
):
    '''
    Get a file with all the parameters from the data and the index where they exceed the threshold.

    Args:
        results_path (str): Path to the data to analyze.

        filepath (str): Path to the file to save the output.

        threshold (float): The threshold to decide if the result is good or not.
    
    Return:
        None
    '''
    data = []
    for folder in track(listdir(results_path), description='Searching best combinations'):
        folder = join(results_path, folder)
        eval_file = join(folder, CaseRun.EVALUATION_FILE.value)
        params_path = join(folder, CaseRun.PARAMS_FILE.value)

        eval_json = load_json(eval_file)
        params = load_json(params_path)
        index = eval_json['nrmse']
        
        data.append({
            'index': index,
            'params': params,
            'folder': folder,
        })

    save_json(data=data, filepath=filepath)


def sort_by_int(array: List):
    '''
    Sort str list as int value.
    '''
    return [str(j) for j in sorted([int(i) for i in array])]


def search_unfinished_combinations(
    path: str,
    depth: int,
    jobs_limit: int,
):
    '''
    Search for the combinations that have not been satisfactorily completed and create a script to execute them.\n
    Args:\n
        path (str): specify the folder where the results of the combinations are stored
        depth (int): depth of the grid
    Return:\n
        None
        
    '''
    info_path = join(path, GridFolders.INFO.value.format(depth=depth))
    comb_path = join(info_path, InfoFiles.COMBINATIONS_FILE.value)
    runs_path = join(path, GridFolders.RUN.value.format(depth=depth), RunFolders.RUN_DATA.value)
    is_valid_file(comb_path, '.json')
    combinations = load_json(comb_path)
    info_data = join(info_path, InfoFiles.INFO_FILE.value)
    params = load_json(info_data)

    unfinished = []
    folders = sort_by_int(listdir(runs_path))
    keys = sort_by_int(combinations.keys())
    for i in track(keys, description='Search unfinished runs'):
        if not folders:
            unfinished.append(i)
        elif i != folders[0]:
            unfinished.append(i)
        else:
            folders.pop(0)
            cpath = join(runs_path, i)
            # TODO: change to cons
            if CaseRun.TIME_FILE.value not in listdir(cpath):
                unfinished.append(i)
                shutil.rmtree(cpath)

    if len(unfinished) == 0:
        print("All combinations terminated")
        return
    print(f"Unfinished combinations: {len(unfinished)}")

    generate_unfinished_script(
        job_name = "unfinished",
        array = compress_numbers(unfinished) ,
        output_path=runs_path,
        data_path = params['data_path'],
        combinations_path = comb_path,
        file_path = join(info_path, InfoFiles.SLURM_UNFINISHED_FILE.value),
        jobs_limit=jobs_limit
    )


def compress_numbers(numbers):
    if not numbers:
        return ""

    result = ""
    start = end = numbers[0]

    for num in numbers[1:]:
        if num == end + 1:
            end = num
        else:
            result += str(start) if start == end else f"{start}-{end}"
            result += ","
            start = end = num

    result += str(start) if start == end else f"{start}-{end}"

    return result


def init_slurm_grid(
    path: str,
    job_name: str,
    jobs_limit: int,
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
    dt: List[float],
    lyapunov_exponent: List[float],
    input_scaling: List[float],
    leak_rate: List[float],
    reservoir_degree: List[int],
    reservoir_sigma: List[float],

    spectral_radius: List[float],
    regularization: List[float],
    rewiring: List[float],
    **kwargs,
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
    params = locals().copy()
    args = params.pop('kwargs')
    params.update(args)
    save_json(params, params_path)

    combinations = generate_combiantions(
        {
            **{
                'units':units,
                'train_length': train_length,
                'forecast_length': forecast_length,
                'transient': transient,
                'steps': steps,
                'dt': dt,
                'lyapunov_exponent': lyapunov_exponent,
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
            },
            **kwargs,
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
        jobs_limit,
    )


def grid_aux(
    path: str,
    n_results: int,
    threshold: float,
):
    '''
    Auxiliar method to use between grid searchs.\n
    It will generate the best results from the previous grid search and generate the new combinations and the new slurm script.

    Args:
        path (str): Path to grid search folders.

        n_results (int): The number of best results to save.

        threshold (float): The threshold to decide if the result is good or not.
    
    Return:
        None
    '''
    n = max([int(i.split('_')[-1]) for i in listdir(path)])

    new_info = join(path, GridFolders.INFO.value.format(depth=n+1))
    new_run = join(path, GridFolders.RUN.value.format(depth=n+1))
    makedirs(new_info, exist_ok=True)
    makedirs(new_run, exist_ok=True)
    combinations_path = join(new_info, InfoFiles.COMBINATIONS_FILE.value)
    new_run_results = join(new_run, RunFolders.RUN_DATA.value)
    script_file = join(new_info, InfoFiles.SLURM_FILE.value)
    results_path = join(path, GridFolders.RUN.value.format(depth=n), RunFolders.RUN_DATA.value)
    output_path = join(new_info, GridFolders.RUN.value.format(depth=n), RunFolders.RUN_RESULTS.value)
    info_data = join(path, GridFolders.INFO.value.format(depth=n), InfoFiles.INFO_FILE.value)

    params = load_json(info_data)

    best_results(
        results_path=results_path,
        output=output_path,
        n_results=n_results,
        threshold=threshold
    )
    # FIX
    new_combinations = generate_result_combinations(
        results_path,
        steps_file,
        new_info,
    )
    generate_slurm_script(
        job_name=params['job_name'],
        array=(1, len(new_combinations)),
        combinations_path=combinations_path,
        output_path=new_run_results,
        data_path=params['data_path'],
        filepath=script_file,
        jobs_limit=params['jobs_limit'],
    )
