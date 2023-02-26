"""Instantiate the different models using function wrappers."""

from tensorflow import keras

from custom_initializers import ErdosRenyi, InputMatrix
from custom_models import ESN, ParallelESN

#### Model instantiators ####


def get_simple_esn(  # Check how to improve this
    units,
    activation="tanh",
    leak_rate=1,
    exponent=2,
    sigma=0.5,
    degree=2,
    spectral_radius=0.99,
    seed=None,
    name="Simple_ESN",
    input_initializer=None,
    reservoir_initializer=None,
    bias_initializer=None,
) -> keras.Model:
    """Get an Ott model. This is an instantiator for the ESN class.

    It is used to make the code more readable.
    It is also used to make the code more flexible. This is easier to use in a grid search.

    Args:
        units (int): The number of units in the reservoir.

        activation (str, optional): The activation function to use in the reservoir.
            Defaults to "tanh".

        leak_rate (int, optional): The leak rate of the reservoir. Defaults to 1.

        exponent (int, optional): The exponent of the activation function. Defaults to 2.

        sigma (float, optional): The sigma of the gaussian distribution to use in the reservoir.
            Defaults to 0.5.

        degree (int, optional): The degree of the polynomial to use in the readout. Defaults to 2.

        spectral_radius (float, optional): The spectral radius of the reservoir. Defaults to 0.99.

        seed (int, optional): The seed to use in the reservoir. Defaults to None.

        name (str, optional): The name of the model. Defaults to "Simple_ESN".

        input_initializer (InputMatrix, optional): The initializer to use in the input matrix.
            Defaults to None.

        reservoir_initializer (ReservoirInitializer, optional): The initializer to use in
            the reservoir matrix. Defaults to None.

        bias_initializer (BiasInitializer, optional): The initializer to use in the bias matrix.
            Defaults to None.

    Returns:
        model: The model to use in the training.
    """
    # These are if the initializer needs to be different than the default.
    if input_initializer is None:
        input_initializer = InputMatrix(sigma=sigma)

    if reservoir_initializer is None:
        reservoir_initializer = ErdosRenyi(
            degree=degree,
            spectral_radius=spectral_radius,
            sigma=spectral_radius,
        )

    if bias_initializer is None:
        bias_initializer = keras.initializers.Zeros()

    model = ESN(
        units=units,
        name=name,
        esn_activation=activation,
        leak_rate=leak_rate,
        seed=seed,
        exponent=exponent,
        input_reservoir_init=input_initializer,
        reservoir_kernel_init=reservoir_initializer,
        input_bias_init=bias_initializer,
    )

    return model


# Using defaults as Ott et al. 2018 KS model
def get_parallel_esn(
    units_per_reservoir,
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
    """Get a parallel ESN model. This is an instantiator for the ParallelESN class.

    It is used to make the code more readable. It is also used to make the code more flexible.
    This is easier to use in a grid search.

    Args:
        units_per_reservoir (int): The number of units per reservoir.

        reservoir_amount (int, optional): The number of reservoirs. Defaults to 64.

        overlap (int, optional): The number of overlapping units between reservoir inputs.
            Defaults to 6.

        leak_rate (int, optional): The leak rate of the reservoirs. Defaults to 1.

        exponent (int, optional): The exponent of the PowerIndex layer (augmented hidden states).
            Defaults to 2.

        sigma (float, optional): The standard deviation of the input matrix. Defaults to 0.5.

        degree (int, optional): The degree of the Erdos-Renyi graph. Defaults to 2.

        spectral_radius (float, optional): The spectral radius of the reservoirs. Defaults to 0.99.

        seed (int, optional): The seed for the random number generator.
            Defaults to None, which means the seed is random.

        name (str, optional): The name of the model. Defaults to "Parallel_ESN".

        input_initializer (InputMatrix, optional): The initializer for the input matrix.
            Defaults to None, which means the default initializer is used.

        reservoir_initializer (ErdosRenyi, optional): The initializer for the reservoir matrix.
            Defaults to None, which means the default initializer is used.

        bias_initializer (keras.initializers.Zeros, optional): The initializer for the bias.
            Defaults to None, Zeros is used.
    """
    # These are if the initializer needs to be different than the default.
    if input_initializer is None:
        input_initializer = InputMatrix(sigma=sigma)

    if reservoir_initializer is None:
        reservoir_initializer = ErdosRenyi(
            degree=degree,
            spectral_radius=spectral_radius,
            sigma=spectral_radius,
        )

    if bias_initializer is None:
        bias_initializer = keras.initializers.Zeros()

    model = ParallelESN(
        units_per_reservoir=units_per_reservoir,
        reservoir_amount=reservoir_amount,
        overlap=overlap,
        name=name,
        esn_activation="tanh",
        leak_rate=leak_rate,
        seed=seed,
        exponent=exponent,
        input_reservoir_init=input_initializer,
        reservoir_kernel_init=reservoir_initializer,
        input_bias_init=bias_initializer,
    )

    return model
