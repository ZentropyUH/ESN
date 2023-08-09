from src.grid.grid_tools import *
import json
from rich.progress import track
from itertools import product, chain


def generate_new_combinations(
        best_path,
        current_path='./',
        intervals_len_file='intervals_len.json',
        output='./combinations.json',
        max_size=2,
        threshold=0.1):
    
    '''using the best combinations of the last run and the intervals len saved'''
    
    best_combinations(path=best_path,
                      output=current_path,
                      max_size=max_size,
                      threshold=threshold)

    combinations = []
    len_intervals = []
    path_len = f"{current_path}{intervals_len_file}"

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

    with open(path_len, 'r+') as f:
        int_len = json.load(f)
        len_intervals = int_len['all']
        len_intervals = [i/10 for i in len_intervals]
        int_len['all'] = len_intervals
    with open(path_len, 'w') as f:
        json.dump(int_len, f)

    new_combinations = []
    # for i in range(len(combinations[0])):
    #     temp = []
    #     for j in range(len(combinations)):
    #         if len(len_intervals) == i+2:
    #             temp.append(combinations[j][i]*len_intervals[i])
    #             temp.append(combinations[j][i]/len_intervals[i])
    #         else:
    #             temp.append(combinations[j][i]+len_intervals[i])
    #             temp.append(combinations[j][i]-len_intervals[i])

    #     new_combinations.append(temp)
    # new_combinations = product(*new_combinations)

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

    with open(join(output), 'w') as data:
        json.dump(new_combinations, data)



