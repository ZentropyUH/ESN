"""Generate the different readout layers for the ESNs."""
import time

import numpy as np
from sklearn.linear_model import ElasticNet, Lasso, Ridge
import keras

from custom_models import ModelWithReadout

######### READOUT GENERATORS #########


def linear_readout(
    model,
    transient_data,
    train_data,
    train_target,
    output_dim=None,
    regularization=1e-8,
    method="ridge",
    solver="svd",  # This solver is the best
) -> keras.Model:
    """Train a linear readout for the given model.

    We are using the Ridge regression from sklearn instead of the keras
    implementation because it is straightforward. The keras implementation is a gradient descent,
    hence an overkill to a linear regression. The svd solver is the most stable and efficient
    solver for the ridge regression with sklearn.

    Args:
        model (keras.Model): The model to be used for the forecast.

        transient_data (np.array): The transient data to be used for the forecast.

        train_data (np.array): The train data to be used for the forecast.

        train_target (np.array): The train target to be used for the forecast.

        regularization (float, optional): The regularization parameter for the Ridge regression.
                                            Defaults to 1e-8.

        method (str, optional): 'ridge' or 'lasso'. The method to be used for the linear readout.
             Defaults to 'ridge'.

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
    if output_dim is None:
        features = train_data.shape[-1]
    else:
        features = output_dim

    readout_layer = keras.layers.Dense(
        units=features, activation="linear", name="readout"
    )

    print()
    print("Harvesting...\n")

    # measure the time of the harvest

    start = time.time()
    # It is better to call model.predict() instead of model()
    # because the former does not compute the gradients. ???
    harvested_states = model.predict(train_data)
    end = time.time()
    print(f"Harvesting took: {round(end - start, 2)} seconds.")

    print()
    print("Harvested states shape: ", harvested_states[0].shape)
    print("Train target shape: ", train_target[0].shape)
    print()

    # Calculating the Tikhnov regularization using sklearn
    print("Calculating the readout matrix...\n")

    if method == "ridge":
        readout = Ridge(alpha=regularization, tol=0, solver=solver)
    elif method == "lasso":
        readout = Lasso(alpha=regularization, tol=0)
    elif method == "elastic":
        readout = ElasticNet(
            alpha=regularization, tol=1e-4, selection="random"
        )
    else:
        raise ValueError("The method must be ['ridge' | 'lasso' | 'elastic'].")

    readout.fit(harvested_states[0], train_target[0])

    # Training error of the readout
    # this is the same as harvested_states[0] @ readout.coef_ + readout.intercept_
    predictions = readout.predict(harvested_states[0])

    training_loss = np.mean((predictions - train_target[0]) ** 2)
    print(f"Training loss: {training_loss}\n")

    # Building the Layer
    readout_layer.build(harvested_states[0].shape)

    # Applying the readout weights
    readout_layer.set_weights([readout.coef_.T, readout.intercept_])

    # WARNING: debuggin this part

    # Adding the readout layer to the model
    # Obscure way to do it but it circumvents the problem of the input
    # being of fixed size. Maybe look into it later.
    out_model = ModelWithReadout(model, readout_layer)

    # Have to build for some reason TODO
    out_model.build(transient_data.shape)

    # Calling the model in order to be able to save it. Check if
    # this is necessary or better ways to do it
    out_model(transient_data[:, :1, :])

    return model


def sgd_linear_readout(
    model,
    transient_data,
    train_data,
    train_target,
    learning_rate=0.001,
    epochs=200,
    regularization=1e-8,
) -> keras.Model:
    """Generate an SGD readout layer.

    Args:
        model (keras.Model): The model without the readout layer.

        transient_data (tf.Tensor): The transient data to ensure ESP on the model.

        train_data (tf.Tensor): The training data.

        train_target (tf.Tensor): The training target

        learning_rate (float, optional): The learning rate for the optimizer. Defaults to 0.001.

        epochs (int, optional): The number of epochs to train. Defaults to 200.

        regularization (float, optional): The regularization parameter of the layer.
            Defaults to 1e-8.

    Returns:
        keras.Model: The model with the readout layer attached.
    """
    print("Training linear readout.")
    print()

    print("Ensuring ESP...\n")  # ESP = Echo State Property

    if not model.built:
        model.build(input_shape=transient_data.shape)

    model.predict(transient_data)

    print()
    print("Harvesting...\n")

    # measure the time of the harvest

    start = time.time()
    # It is better to call model.predict() instead of model()
    # because the former does not compute the gradients. ???
    harvested_states = model.predict(train_data)
    end = time.time()
    print(f"Harvesting took: {round(end - start, 2)} seconds.")

    print()
    print("Harvested states shape: ", harvested_states[0].shape)
    print("Train target shape: ", train_target[0].shape)
    print()

    features = harvested_states.shape[-1]

    output_dim = train_target.shape[-1]

    inputs = keras.Input(shape=(None, features))
    readout_layer = keras.layers.Dense(
        units=output_dim,
        activation="linear",
        name="readout_layer",
        kernel_regularizer=keras.regularizers.l2(regularization),
    )(inputs)

    readout = keras.Model(inputs=inputs, outputs=readout_layer, name="readout")

    readout.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.MeanSquaredError(),
    )

    print("Training the readout...\n")

    readout.fit(
        x=harvested_states[0],
        y=train_target[0],
        epochs=epochs,
        callbacks=None,
        verbose=1,
        validation_split=0.2,
    )

    # Adding the readout layer to the model
    # Obscure way to do it but it circumvents the problem of the input
    # being of fixed size. Maybe look into it later.
    model = ModelWithReadout(model, readout.get_layer("readout_layer"))

    # Have to build for some reason TODO
    model.build(transient_data.shape)

    # Calling the model in order to be able to save it TODO: Check if
    # this is necessary or better ways to do it
    model.predict(transient_data[:, :1, :], verbose=0)

    return model
