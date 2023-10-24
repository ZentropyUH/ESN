import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Any
from typing import List
from typing import Dict
from typing import Tuple
from itertools import combinations


def _group_results(group_params: List[str], data: Dict[str, Any], regroup_param: str = None):
    '''
    Given the data and the params to group, return a dict with the index of the data
    
    Args:
        group_params (List[str]): List of params to group.
        
        data (Dict): Data to group. The data must have a 'params' key with the params to group and a 'index' key with the index where the data exceed the threshold.
        
        regroup_param (str): param to regroup the data.
        
    Returns:
        Dict: dict with the index of the data.
    '''
    param_data = {}
    for x in data:
        # get tuple of especific params
        current_params = tuple((x['params'][a] for a in group_params))
        
        # insert index value in dict param_data by params tuple
        if param_data.get(current_params):
            param_data[current_params].append(x['index'])
        else:
            param_data[current_params] = [x['index']]
    
    # to regroup by a param
    if regroup_param:
        new_data = {}
        index = group_params.index(regroup_param)
        for key in param_data.keys():
            new_key = list(key)
            k = new_key.pop(index)
            new_key = tuple(new_key)
            
            if new_data.get(k):
                new_data[k][new_key] = param_data[key]
            else:
                new_data[k] = {new_key: param_data[key]}
        return new_data

    return param_data


def _apply_functions(data: Dict[str, List[float]]):
    '''
    Calculate the mean and the std of every param combination.

    Args:
        data (Dict[str, List[float]]): dict with the index of the data.
    
    Returns:
        Dict[str, Dict[str, float]]: dict with the mean and the std of every param combination.
    '''
    # save the mean and the std of every param combination
    new_data = {}
    for i in data:
        elements = np.array(data[i])
        new_data[i] = {'mean': elements.mean(), 'std': elements.std()}
    return new_data


def _generate_meshgrid(data: Dict[Tuple, float]):
    '''
    Generate the meshgrid of the data.
    
    Args:
        data (Dict[Tuple, float]): dict with the mean and the std of every param combination.
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: meshgrid of the data.
    '''
    X = set()
    Y = set()
    for i in data.keys():
        X.add(i[0])
        Y.add(i[1])
    X, Y = np.meshgrid(sorted(list(X)), sorted(list(Y)))

    Z = []
    STD = []
    for x, y in zip(X, Y):
        z = []
        std = []
        for _x, _y in zip(x,y):
            z.append(data[(_x,_y)]['mean'])
            std.append(data[(_x,_y)]['std'])
        Z.append(z)
        STD.append(std)
    Z = np.array(Z)
    
    return X, Y, Z, STD


def _plot_3d_std(X, Y, Z, STD, params: List[str]):
    fig = plt.figure(dpi=100)
    ax = plt.axes(projection ='3d')
    
    ax.plot_wireframe(X, Y, Z, edgecolor='green')
    
    for x, y, z, std in zip(X, Y, Z, STD):
        for _x, _y, _z, _std in zip(x, y, z, std):
            ax.plot([_x, _x], [_y, _y], [_z+_std, _z-_std], marker='_', color='red')

    ax.set_title('Title')
    ax.set_xlabel(params[0], fontsize=12)
    ax.set_ylabel(params[1], fontsize=12)
    ax.set_zlabel('z', fontsize=12)
    plt.show()
    plt.close()


def _plot_3d(X, Y, Z, params: List[str], title: str):
    fig = plt.figure(figsize=(12, 4))
    ax = plt.axes(projection ='3d')
    
    ax.plot_surface(X, Y, Z, cmap='inferno', edgecolor='grey', rstride=1, cstride=1)

    ax.set_title(title)
    ax.set_xlabel(params[0], fontsize=12)
    ax.set_ylabel(params[1], fontsize=12)
    ax.set_zlabel('z', fontsize=12)
    plt.show()
    plt.close()


def _plot_grouped_3d(data: Dict, params: List[str], title: str, grouped_param: str, std: bool = False):
    fig = plt.figure(figsize=(20, 8))
    axs = [fig.add_subplot(1, len(data), i+1, projection='3d') for i, _ in enumerate(range(len(data)))]

    for i, key in enumerate(sorted(data.keys())):
        X, Y, Z, _ = data[key]
        axs[i].plot_surface(X, Y, Z, cmap='magma', edgecolor='grey', rstride=1, cstride=1)

        axs[i].set_title(grouped_param + f': {str(key)}', fontsize=12)
        axs[i].set_xlabel(params[0], fontsize=12)
        axs[i].set_ylabel(params[1], fontsize=12)
        axs[i].set_zlabel('z', fontsize=12)
    
    plt.show()
    plt.close()

    if std:
        fig = plt.figure(figsize=(20, 8))
        axs = [fig.add_subplot(1, len(data), i+1, projection='3d') for i, _ in enumerate(range(len(data)))]

        for i, key in enumerate(sorted(data.keys())):
            X, Y, Z, STD = data[key]

            axs[i].plot_wireframe(X, Y, Z, edgecolor='green')
            for x, y, z, std in zip(X, Y, Z, STD):
                for _x, _y, _z, _std in zip(x, y, z, std):
                    axs[i].plot([_x, _x], [_y, _y], [_z+_std, _z-_std], marker='_', color='red')

            axs[i].set_title(grouped_param + f': {str(key)}', fontsize=12)
            axs[i].set_xlabel(params[0], fontsize=12)
            axs[i].set_ylabel(params[1], fontsize=12)
            axs[i].set_zlabel('z', fontsize=12)
        
        plt.show()
        plt.close()


def plot_params_by(
    datapath: str,
    params: List[str],
    by_param: str
):
    cases = list(combinations(params, 2))
    with open(datapath, 'r') as f:
        data: Dict = json.load(f)
    
    for case in cases:
        grouped_data = _group_results(list(case)+[by_param], data, 'reservoir_degree')
        _data = {i: _generate_meshgrid(_apply_functions(grouped_data[i])) for i in sorted(grouped_data.keys())}
        _plot_grouped_3d(_data, case, by_param, by_param, True)


def plot_params(
    datapath: str,
    params: List[str],
):
    cases = list(combinations(params, 2))
    with open(datapath, 'r') as f:
        data: Dict = json.load(f)

    for case in cases:
        grouped_data = _group_results(case, data)
        grouped_data = _apply_functions(grouped_data)
        X, Y, Z, STD = _generate_meshgrid(grouped_data)
        _plot_3d(X, Y, Z, case, 'Title')
        _plot_3d_std(X, Y, Z, STD, case)