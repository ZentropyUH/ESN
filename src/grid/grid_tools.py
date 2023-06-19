from os import makedirs, listdir, system
from os.path import join, isdir, split

from threading import Thread
from random import randint
from itertools import product

import pandas as pd
import numpy as np
import csv
import json
import shutil
import matplotlib.pyplot as plt

from model_functions import _train, _forecast


# Priority Queue with limited size, sorted from max to min
class Queue:
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


def save_plots(data: list, output_path: str, name: str):
    plt.clf()
    plt.figure()
    plt.plot(data)
    plt.xlabel("Time")
    plt.ylabel("Mean square error")
    plt.title("Plot of root mean square error")
    plt.savefig(join(output_path, name))


def save_plots_from_csv(input_path, output_path, name):
    with open(input_path, "r") as archivo:
        data = list(csv.reader(archivo))

    data = [float(date[0]) for date in data]

    plt.clf()
    plt.figure()
    plt.plot(data)
    plt.xlabel("Time")
    plt.ylabel("Mean square error")
    plt.title("Plot of root mean square error")
    plt.savefig(output_path + name)


def save_csv(data, name: str, path: str):
    pd.DataFrame(data).to_csv(
        join(path, name),
        index=False,
        header=None,
    )


def save_plots_data_forecast(data: str, forecast: str, output: str):
    plt.clf()
    plt.figure()
    plt.plot(data, color="blue")
    plt.plot(forecast, color="red")
    plt.xlabel("Time")
    plt.ylabel("value")
    plt.title("Plot data and forecast")
    plt.savefig(output)


def get_axis(data, axis: int):
    return [x[axis] for x in data]


def plots_data_forecast(data: str, forecast: str, output: str):
    save_plots_data_forecast(
        get_axis(data, 0), get_axis(forecast, 0), output + "_x"
    )
    save_plots_data_forecast(
        get_axis(data, 1), get_axis(forecast, 1), output + "_y"
    )
    save_plots_data_forecast(
        get_axis(data, 2), get_axis(forecast, 2), output + "_z"
    )


def read_csv(file: str):
    return pd.read_csv(file).to_numpy()


def generate_combinations(params: dict):
    return [
        [elem[3](elem[0], elem[2], i) for i in range(elem[1])]
        for elem in params.values()
    ]


def get_param_tuple(value, param, step):
    initial_value, number_of_values, _, function_of_increment = param
    initial_value = value - int(number_of_values / 2) * step
    return initial_value, number_of_values, step, function_of_increment


def get_ritch_param_tuple(value, param, step):
    initial_value, number_of_values, _, function_of_increment = param
    initial_value = value / int(number_of_values / 2) * step
    return initial_value, number_of_values, step, function_of_increment


def calculate_aprox_time(time: list, file: str, text):
    with open(file, "a+") as f:
        f.write("{}: {}\n".format(text, str(np.mean(time))))


def save_combinations_txt(hyperparameters_to_adjust: dict, path: str):
    with open(join(path, "combinations.txt"), "w") as f:
        for i, c in enumerate(
            product(*generate_combinations(hyperparameters_to_adjust))
        ):
            f.write(
                "{} {} {} {} {} {}\n".format(
                    i + 1, c[0], c[1], c[2], c[3], c[4]
                )
            )


def save_combinations(hyperparameters_to_adjust: dict):
    with open("./combinations.json", "w") as f:
        json.dump(
            {
                int(i + 1): c
                for i, c in enumerate(
                    product(*generate_combinations(hyperparameters_to_adjust))
                )
            },
            f,
            indent=4,
            sort_keys=True,
            separators=(",", ": "),
        )


def change_folders(path: str):
    for folder in listdir(path):
        folder = join(path, folder)
        time_file = join(folder, "time.txt")

        inside_folders = [
            join(folder, f)
            for f in listdir(folder)
            if join(folder, f) != time_file
        ]

        for inside_folder in inside_folders:
            shutil.move(time_file, inside_folder)
            shutil.move(inside_folder, path)
            shutil.rmtree(folder)


def detect_not_fished_jobs(path: str, output: str):
    with open(join(output, "out.out"), "w") as f:
        for file in [
            join(path, f)
            for f in listdir(path)
            if "time.txt" not in listdir(join(path, f))
        ]:
            f.write("{}\n".format(file.split("_")[-1]))


# main forecast
def forecast_main(
    prediction_steps: int,
    trained_model_path: str,
    prediction_path: str,
    data_file: str,
    forecast_name: str,
):
    instruction = f"python3 ./tmain.py forecast \
            -fm classic \
            -fl {prediction_steps} \
            -tm {trained_model_path} \
            -df {data_file} \
            -o {prediction_path} \
            -fn {forecast_name}"

    system(instruction)


# main plot
def plot_main(prediction_file: str, data_file: str, tl: int, output_file: str):
    instruction = f"python3 ./tmain.py plot \
            -dt 1 \
            -pt linear \
            -tl {tl} \
            -df {data_file} \
            -pr {prediction_file} \
            --save-path {output_file} \
            "

    system(instruction)
