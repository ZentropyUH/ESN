from src.grid.grid_tools import *
import json
from rich.progress import track
from itertools import product, chain


def generate_new_combinations(
        best_path,
        intervals_len_file='./intervals_len.json',
        output='./combinations.json',
):
    
    '''using the best combinations of the last run and the intervals len saved'''

    combinations = []
    len_intervals = []

    for folder in track(listdir(best_path), description='Searching best combinations'):
        folder = join(best_path, folder)
        params_path = join(folder, 'trained_model', 'params.json')

        params = {}
        with open(params_path, 'r') as f:
            params = json.load(f)

        combinations.append(
            [params['reservoir_sigma'],
             params['reservoir_degree'],
             params['regularization'],
             params['spectral_radius'],
             params['rewiring']]
        )

    with open(intervals_len_file, 'r+') as f:
        int_len = json.load(f)
        len_intervals = int_len['all']
        len_intervals = [i/10 for i in len_intervals]
        int_len['all'] = len_intervals
    with open(intervals_len_file, 'w') as f:
        json.dump(int_len, f)

    new_combinations = []

    for i in range(len(combinations)):       
        new_combinations.append([]) 
        for j in range(len(combinations[i])):
            if len(len_intervals) == j+2:
                new_combinations[i].append(combinations[i][j]*len_intervals[i])
                new_combinations[i].append(combinations[i][j]/len_intervals[i])
            else:
                new_combinations[i].append(combinations[i][j]+len_intervals[i])
                new_combinations[i].append(combinations[i][j]-len_intervals[i])
        new_combinations[i] = product(*new_combinations[i])
            

    new_combinations= chain(*new_combinations)
    new_combinations = {str(i): x for i, x in enumerate(k for k in new_combinations)}

    with open(output, 'w') as data:
        json.dump(new_combinations, data)



