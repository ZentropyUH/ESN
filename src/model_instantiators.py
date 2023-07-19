"""Instantiate the different models using function wrappers."""

import keras
import keras.layers
import numpy as np
import tensorflow as tf
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


from src.customs.custom_models import ParallelESN
from src.customs.custom_layers import EsnCell, PowerIndex

#### Model instantiators ####


def create_esn_model(
    # ESN related parameters
    features=1,
    units=200,
    leak_rate=1,
    input_reservoir_init="InputMatrix",
    input_bias_init="random_uniform",
    reservoir_kernel_init="WattsStrogatzNX",
    esn_activation="tanh",
    exponent=2,
    # Seed of the model
    seed=None,
    regularization=1e-8,
):
    """
    Create an ESN model using Keras' functional API.

    Args:
        units (int): Number of units in the reservoir.

        leak_rate (float): Leaky rate of the reservoir.

        input_reservoir_init (str): Initializer for the input to reservoir weights.
            Defaults to 'InputMatrix'.

        input_bias_init (str): Initializer for the input bias.
            Defaults to 'random_uniform'.

        reservoir_kernel_init (str): Initializer for the reservoir weights.
            Defaults to 'ErdosRenyi'.

        esn_activation (str): Activation function of the reservoir.
            Defaults to 'tanh'.

        exponent (int): Exponent of the power function applied to the reservoir.
            Defaults to 2.

        seed (int, optional): Seed of the model. If None, a random seed will be used.

    Returns:
        keras.Model: A Keras model with the ESN architecture.

    Example usage:

    >>> model = create_esn_model(
            units=100,
            leak_rate=0.5,
            input_reservoir_init='glorot_uniform',
            input_bias_init='zeros',
            reservoir_kernel_init='ErdosRenyi',
            esn_activation='tanh',
            exponent=2,
            seed=42,
        )
    """
    # Optional seed of the model
    if seed is None:
        seed = np.random.randint(0, 1000000)

    print()
    print(f"Seed: {seed}\n")

    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Define the input layer
    inputs = keras.Input(batch_shape=(1, None, features), name="Input")

    # Define the ESN cell layer
    esn_cell = EsnCell(
        units=units,
        name="EsnCell",
        activation=esn_activation,
        leak_rate=leak_rate,
        input_initializer=input_reservoir_init,
        input_bias_initializer=input_bias_init,
        reservoir_initializer=reservoir_kernel_init,
    )

    # Define the RNN layer using the ESN cell
    esn_rnn = keras.layers.RNN(
        esn_cell,
        trainable=False,
        stateful=True,
        return_sequences=True,
        name="esn_rnn",
    )(inputs)

    # Define the PowerIndex layer
    power_index = PowerIndex(exponent=exponent, index=2, name="power_index")(
        esn_rnn
    )

    output = keras.layers.Concatenate(name="Concat_ESN_input")(
        [inputs, power_index]
    )

    # Build the model
    model = keras.Model(
        inputs=inputs, outputs=output, name="ESN_without_readout"
    )

    # Return the created model
    return model


# Using defaults as Ott et al. 2018 KS model
def get_parallel_esn(
    features=1,
    units_per_reservoir=200,
    reservoir_amount=64,
    overlap=6,
    leak_rate=1,
    exponent=2,
    sigma=0.5,
    degree=2,
    spectral_radius=0.99,
    seed=None,
    name="Parallel_ESN",
    input_initializer=None,
    reservoir_initializer=None,
    bias_initializer=None,
) -> keras.Model:
    if seed is None:
        seed = np.random.randint(0, 1000000)

    print()
    print(f"Seed: {seed}\n")

    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Define the input layer
    inputs = keras.Input(batch_shape=(1, None, features), name="Input")
