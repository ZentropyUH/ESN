from src.grid.grid_script import *
import src.grid.grid_script
import src.grid.grid_script_threads


# The hyperparameters will be of the form: name: (initial_value, number_of_values, increment, function_of_increment)
# The parameters of the increment function are: initial_value, increment, current_value_of_the_iteration
hyperparameters_to_adjust = {
    "sigma": (0.2, 5, 0.2, lambda x, y, i: round(x + y * i, 2)),
    "degree": (2, 4, 2, lambda x, y, i: round(x + y * i, 2)),
    "ritch_regularization": (10e-5, 5, 0.1, lambda x, y, i: x * y**i),
    "spectral_radio": (0.9, 16 , 0.01, lambda x, y, i: round(x + y * i, 2)),
    "reconection_prob": (0, 6, 0.2, lambda x, y, i: round(x + y*i, 2))
}

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparametes")
    parser.add_argument('-t', '--type', help="Grid type", type=str, required=True, choices=['f_threads', 'grid'], default='grid')
    parser.add_argument('-o', '--output', help="Output path", type=str, required=True)
    parser.add_argument('-d', '--data', help="Data path", type=str, required=True)
    parser.add_argument('-n', help="Search Tree depth", type=int, default=5)
    parser.add_argument('-m', help="Number of best to keep", type=int, default=2)
    args = parser.parse_args()

    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(gpus))
    print(gpus)


    if args.type == 'grid':
        src.grid.grid_script.grid_search(hyperparameters_to_adjust, args.data, args.output, args.n, args.m)
    if args.type == 'f_threads':
        src.grid.grid_script_threads.grid_search(hyperparameters_to_adjust, args.data, args.output, args.n, args.m)
