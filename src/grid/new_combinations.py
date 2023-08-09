from src.grid.grid_tools import *
# import time
# import tensorflow as tf
# import argparse
import json
from rich.progress import track
# from model_functions import _forecast, _train, _plot
# import os
# from src.grid.grid_tools import best_combinations
from itertools import product




def generate_new_combinations(
        best_path,
        current_path = './',        
        # data_file = 'combination_threshold_time.json',
        count = 5,
        intervals_len_file = 'intervals_len.json',
        output = 'combinations.json',
        max_size = 2,
        threshold=0.1):
    
    '''using the best combinations of the last run and the intervals len saved'''
    best_combinations(path=best_path,
                      output=current_path,
                      max_size=max_size,
                      threshold=threshold)
    
    combinations = []
    len_intervals = []
    # path = join(current_path, "new_params.json")
    new_params = []
    path_len = f"{current_path}{intervals_len_file}"

    for folder in track(listdir(best_path), description='Searching best combinations'):
        folder = join(best_path, folder)
        # rmse_mean_path = join(folder, 'rmse_mean', 'rmse_mean.csv')
        params_path = join(folder, 'trained_model', 'params.json')

        # rmse_mean = []
        # with open(rmse_mean_path, 'r') as f:
        #     rmse_mean = read_csv(f)
        
        params = {}
        with open(params_path, 'r') as f:
            params = json.load(f)
        
        new_params.append(
            (params['reservoir_sigma'],
            params['reservoir_degree'],
            params['regularization'],
            params['spectral_radius'],
            params['rewiring'])
        )
        
        
    #     best.decide(rmse_mean, params, folder, threshold)
    
    # for i, element in enumerate(best.queue):
    #     folder = element[1][1]
    #     shutil.copytree(folder, join(output, str(i)), dirs_exist_ok=True)

    
    # with open(path, 'r') as f:
    #     data = json.load(f)
    #     data = sorted(data, key=lambda x: x[1])
    #     for i in range(min(count, len(data))):
    #         combinations.append(data[i][0])

    with open(path_len, 'r+') as f:
        int_len = json.load(f)
        len_intervals = int_len['all']
        len_intervals = [i/10 for i in len_intervals]
        int_len['all'] = len_intervals
    with open(path_len, 'w') as f:
        json.dump(int_len, f)

    new_combinations = []
    for i in range(len(combinations[0])):
        temp = []
        for j in range(len(combinations)):
            if len(len_intervals) == i+2:
                temp.append(combinations[j][i]*len_intervals[i])
                temp.append(combinations[j][i]/len_intervals[i])
            else:
                temp.append(combinations[j][i]+len_intervals[i])
                temp.append(combinations[j][i]-len_intervals[i])

        new_combinations.append(temp)
    new_combinations = product(*new_combinations)
    new_combinations = {str(i): x for i, x in enumerate(new_combinations)}

    with open(join( output), 'w') as data:
        json.dump(new_combinations, data)
