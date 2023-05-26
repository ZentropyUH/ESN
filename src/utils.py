"""Define some general utility functions."""

import numpy as np
import pandas as pd
import re
import json

#### Parameters ####


def get_smallest_digit_unit(number):
    """Get the smallest order of magnitude unit of a number.

    Args:
        number (float): The number to be analyzed.

    Returns:
        float: The smallest order of magnitude of the number.
    """
    number = float(number)
    # Convert it to a string
    number = str(number)
    number = number.split(".")
    # Now count the number of digits after the decimal point
    if len(number) == 2 and number[-1] == "0":
        return 1
    else:
        return 10 ** (-len(number[1]))


def get_range(rng, method="linear", step=None, base=10, dtype=float):
    """Get the ranges for the parameters.

    The parameter rng is a string of the form 'start:end' or a list of comma separated values.

    Args:
        rng (str): The range of the parameter. Can be a list of comma separated values or a range
            of the form 'start:end'.

        method (int): The method to be used to generate the range. It can be either 'linear' or 'log'.
            It will generate a range of values between the start and end values,
            linearly with a step of the lowest significant digit of the start value or
            logarithmically spaced with base 10.

        step (float): The step to be used for the linear range. If None, it will be calculated
            automatically.

        base (int): The base to be used for the logarithmic range.

        dtype (type): The data type for the values.

    Returns:
        list: The list of values for the parameter.
    """
    # Check if range
    if ":" in rng:
        start, end = rng.split(":")
        start = float(start)
        end = float(end)
        if method == "linear":
            if step is None:
                step = get_smallest_digit_unit(start)

            return np.arange(start, end + 1, step)  # the +1 to include the end
        elif method == "log":
            start = np.log(start) / np.log(base)
            end = np.log(end) / np.log(base)
            print(start, end)
            exps = np.arange(start, end + np.sign(end), np.sign(end))
            print(exps)
            return (base**exps).astype(dtype)
    # Check if comma separated values
    elif "," in rng:
        return np.array([float(item) for item in rng.split(",")]).astype(dtype)
    # Check if single value
    else:
        try:
            return np.array([float(rng)]).astype(dtype)
        except ValueError:
            print(
                "The units parameter should be a single value, a range or a list of values"
            )


def lyap_ks(i, l):
    """Estimation of the i-th largest Lyapunov Time of the KS model.

    Taken from the paper:
        "Lyapunov Exponents of the Kuramoto-Sivashinsky PDE. arxiv:1902.09651v1"
    """
    # This approximation is taken from the above paper. Verify veracity.
    return 0.093 - 0.94 * (i - 0.39) / l


def get_name_from_dict(dictionary):
    """Output a string with each key concatenated with its value.

    The elements are ordered lexicografically by the keys.

    Args:
        dictionary (dict): The dictionary to be converted to a string.
    Returns:
        str: The string with the concatenated keys and values.
    """
    concatenated_str = "-".join(
        [f"{key}_'{dictionary[key]}'" for key in sorted(dictionary.keys())]
    )
    return concatenated_str


def get_dict_from_name(name_str):
    """Return a dictionary from a string with each key and value.

    The elements are extracted using regular expressions.

    Args:
        name_str (str): The string to be converted to a dictionary.
    Returns:
        dict: The dictionary with the extracted keys and values.
    """
    regex = r"([a-zA-Z_]+)_\'([^\']+)\'"
    matches = re.findall(regex, name_str)
    return {key: value for key, value in matches}


def load_data(
    name: str,
    init_transient: int = 0,
    transient: int = 1000,
    train_length: int = 5000,
    step: int = 1,
):
    """Load the data from the given path. Returns a dataset for training a NN.

    Data is supposed to be stored in a .csv and has a shape of (T, D), (T)ime and (D)imensions.

    Args:
        name (str): The name of the file to be loaded.

        transient (int, optional): The length of the training transient
                                    for teacher enforced process. Defaults to 1000.

        train_length (int, optional): The length of the training data. Defaults to 5000.

        init_transient (int, optional): The initial transient of the data. Defaults to 0.
            This amount of initial values are to be ignored to ensure stationarity of the system.

        step: Sets the number of steps between data sampling. i. e. takes values every 'step' steps

    Returns:
        tuple: A tuple with:

                transient_data: The transient of the training data. This is to ensure ESP.

                training_data: Training data.

                training_target: The training target. This is for forecasting, so target data is
                    the training data taken shifted 1 index to the right plus one value.

                forecast_transient_data: The last 'transient' elements in training_data.
                    This is to ensure ESP.

                validation_data: Validation data

                validation_target: The validation target. This is for forecasting, so target data is
                    the validation data taken shifted 1 index to the right plus one value.
    """
    data = pd.read_csv(name).to_numpy()

    # Take the elements of the data skipping every step elements.
    data = data[::step]

    if step > 1:
        print(
            "Used data shape: ",
            data.shape,
            f"Picking values every {step} steps.",
        )

    # Check the columns of the data (D)
    features = data.shape[-1]

    # Reshape it to have batches as a dimension. For keras convention purposes.
    data = data.reshape(1, -1, features)

    # Ignoring initial transient
    data = data[:, init_transient:, :]

    # Index up to the training end.
    train_index = transient + train_length

    if train_index > data.shape[1]:
        raise ValueError(
            f"The train size is out of range. Data size is: "
            f"{data.shape[1]} and train size + transient is: {train_index}"
        )

    # Transient data (For ESP purposes)
    transient_data = data[:, :transient, :]

    train_data = data[:, transient:train_index, :]
    train_target = data[:, transient + 1 : train_index + 1, :]

    # Forecast transient (For ESP purposes).
    # These are the last 'transient' values of the training data
    forecast_transient_data = train_data[:, -transient:, :]

    val_data = data[:, train_index:-1, :]
    val_target = data[:, train_index + 1 :, :]

    return (
        transient_data,
        train_data,
        train_target,
        forecast_transient_data,
        val_data,
        val_target,
    )


def load_model_json(model_path):
    with open(model_path + "/params.json", encoding="utf-8") as f:
        data = json.load(f)
    return data


# Decorator for composing a function n times.
def compose_n_times(n):
    """Compose a function n times."""

    def decorator(f):
        def inner(x):
            result = x
            for i in range(n):
                result = f(result)
            return result

        return inner

    return decorator


if __name__ == "__main__":

    def f(x):
        return x + 1

    print(compose_n_times(10)(f)(0))
