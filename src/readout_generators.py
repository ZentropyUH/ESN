"""Generate the different readout layers for the ESNs."""
import time

import numpy as np
import keras
from src.utils import tf_ridge_regression


######### READOUT GENERATORS #########


def linear_readout(
    model,
    transient_data,
    train_data,
    train_target,
    regularization=1e-8,
    # solver="svd",  # This solver is the best
) -> keras.Model:
    """Train a linear readout for the given model.

        We are using the Ridge regression from sklearn instead of the keras
        implementation because it is straightforward. The keras implementation is a gradient descent,
        hence an overkill to a linear regression. The svd solver is the most stable and efficient
        solver for the ridge regression with sklearn.

        Args:
            model (keras.Model): The model to be used for the forecast.
    =
            transient_data (np.array): The transient data to be used for the forecast.

            train_data (np.array): The train data to be used for the forecast.

            train_target (np.array): The train target to be used for the forecast.

            regularization (float, optional): The regularization parameter for the Ridge regression.
                                                Defaults to 1e-8.

            solver (str, optional): Only when method='ridge'
                The solver to be used for the linear readout. Defaults to "svd".

        Returns:
            model (keras.Model): The model with the readout layer attached.
    """
    print("Training linear readout.")
    print()

    print("Ensuring ESP...\n")  # ESP = Echo State Property

    if not model.built:
        model.build(input_shape=transient_data.shape)

    model.predict(transient_data)

    # Creating the readout layer
    features = train_data.shape[-1]

    print()
    print("Harvesting...\n")

    # measure the time of the harvest

    start = time.time()
    # It is better to call model.predict() instead of model()
    # because the former does not compute the gradients.
    harvested_states = model.predict(train_data)
    end = time.time()
    print(f"Harvesting took: {round(end - start, 2)} seconds.")

    print()
    print("Harvested states shape: ", harvested_states.shape)
    print("Train target shape: ", train_target.shape)
    print()

    # Calculating the Tikhnov regularization using sklearn
    print("Calculating the readout matrix...\n")

    readout_matrix, readout_bias = tf_ridge_regression(
        harvested_states[0], train_target[0], regularization
    )

    readout_layer = keras.layers.Dense(
        features, activation="linear", name="readout", trainable=False
    )

    # Building the Layer
    readout_layer.build(harvested_states[0].shape)

    # Applying the readout weights
    readout_layer.set_weights([readout_matrix, readout_bias])

    # readout = Ridge(alpha=regularization, tol=0, solver=solver)

    # readout.fit(harvested_states[0], train_target[0])

    # Training error of the readout
    predicted = readout_layer(harvested_states[0])

    training_loss = np.mean((predicted - train_target[0]) ** 2)

    print(f"Training loss: {training_loss}\n")

    model = keras.Model(
        inputs=model.inputs,
        outputs=readout_layer(model.outputs),
        name="ESN",
    )

    return model
