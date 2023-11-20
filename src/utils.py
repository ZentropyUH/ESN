"""Define some general utility functions."""

import json
import re

import numpy as np
import pandas as pd


#### Parameters ####

# given i it starts from letter x and goes cyclically, when x reached starts xx, xy, etc.
letter = lambda n: 'x' * ((n + 23) // 26) + chr(ord('a') + (n + 23) % 26)

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

    features = data.shape[-1]

    data = data[::step]

    data = data.reshape(1, -1, features)

    # Take the elements of the data skipping every step elements.

    if step > 1:
        print(
            "Used data shape: ",
            data.shape,
            f"Picking values every {step} steps.",
        )

    # Index up to the training end.
    train_index = transient + train_length

    if train_index > data.shape[1]:
        raise ValueError(
            f"The train size is out of range. Data size is: "
            f"{data.shape[0]} and train size + transient is: {train_index}"
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


def load_model_and_params(model_path: str):
    from keras.models import load_model
    # Load the param json from the model location
    with open(model_path + "/params.json", encoding="utf-8") as f:
        params = json.load(f)

    model = load_model(model_path, compile=False)

    return model, params


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


# Get the state of the ESN function
def get_esn_state(model):
    """Return the state of the ESN cell.

    Args:
        model (Model): The Keras model containing the ESN RNN layer.

    Returns:
        np array
    """
    # Access the ESN RNN layer by name and retrieve its last state
    esn_rnn_layer = model.get_layer("esn_rnn")
    state_h = esn_rnn_layer.states[0]

    # Convert the tensor to a NumPy array
    states = np.squeeze(state_h.numpy())

    return states
