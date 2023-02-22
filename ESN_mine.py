"""Do main stuff here. Correct docstring later."""
import os
import time
import timeit

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from matplotlib.animation import FuncAnimation

# from scipy.signal import (  # This is to have relative maximums and minimums.
#     argrelmax,
#     argrelmin,
# )
from sklearn.linear_model import Ridge, ElasticNet, Lasso, SGDRegressor
from tensorflow import keras
from tqdm import trange

from custom_models import ESN, ModelWithReadout, ParallelESN
from custom_initializers import (
    InputMatrix,
    RegularOwn,
    ErdosRenyi,
    WattsStrogatzOwn,
    RegularNX,
)

# To avoid tensorflow verbosity
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# To use the CPU instead of the GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = ""


print("\n\n")

#### Parameters ####


def lyap_ks(i, L):
    """Estimation of the i-th largest Lyapunov Time of the KS model.

    Taken from the paper:
        "Lyapunov Exponents of the Kuramoto-Sivashinsky PDE"
    """
    return 0.093 - 0.94 * (i - 0.39) / L


def load_data(
    name: str,
    transient: int = 1000,
    train_length: int = 5000,
    init_transient: int = 0,
    step: int = 1,
) -> tuple:
    """Load the data from the given path. Returns a dataset for training a NN.

    Args:
        name (str): The name of the file to be loaded.

        transient (int, optional): The length of the training transient
                                    for teacher enforced process. Defaults to 1000.

        train_length (int, optional): The length of the training data. Defaults to 5000.

        init_transient (int, optional): The initial transient to ignore from the data.
                                            Defaults to 0.

    Returns:
        tuple: A tuple with:
                transient data,\n
                training data, \n
                training target,\n
                forecast transient data, \n
                validation data,\n
                validation target.
    """
    data = pd.read_csv(name).to_numpy()

    print("Original data shape: ", data.shape)
    # Take the elements of the data skipping every step elements.
    data = data[::step]

    if step > 1:
        print(
            "Used data shape: ",
            data.shape,
            f"Picking values every {step} steps",
        )

    # Check the columns of the data
    features = data.shape[-1]

    data = data.reshape(1, -1, features)

    data = data[:, init_transient:, :]

    print(f"data shape: {data.shape}")

    train_index = transient + train_length

    if train_index > data.shape[1]:
        raise ValueError(
            f"The train size is out of range. Data size is: {data.shape[1]} and train size is: {train_index}"
        )

    transient_data = data[:, :transient, :]
    train_data = data[:, transient:train_index, :]
    train_target = data[:, transient + 1 : train_index + 1, :]

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


#### Model functions ####


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
):
    """Get an Ott model. This is a wrapper for the ESN class. It is used to make the code more readable. It is also used to make the code more flexible. This is easier to use in a grid search.

    Args:
        units (int): The number of units in the reservoir.

        activation (str, optional): The activation function to use in the reservoir. Defaults to "tanh".

        leak_rate (int, optional): The leak rate of the reservoir. Defaults to 1.

        exponent (int, optional): The exponent of the activation function. Defaults to 2.

        sigma (float, optional): The sigma of the gaussian distribution to use in the reservoir. Defaults to 0.5.

        degree (int, optional): The degree of the polynomial to use in the readout. Defaults to 2.

        spectral_radius (float, optional): The spectral radius of the reservoir. Defaults to 0.99.

        seed (int, optional): The seed to use in the reservoir. Defaults to None.

        name (str, optional): The name of the model. Defaults to "Simple_ESN".

        input_initializer (InputMatrix, optional): The initializer to use in the input matrix. Defaults to None.

        reservoir_initializer (ReservoirInitializer, optional): The initializer to use in the reservoir matrix. Defaults to None.

        bias_initializer (BiasInitializer, optional): The initializer to use in the bias matrix. Defaults to None.

    Returns:
        model: The model to use in the training.
    """
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
):
    """Get a parallel ESN model. This is a wrapper for the ParallelESN class. It is used to make the code more readable. It is also used to make the code more flexible. This is easier to use in a grid search.

    Args:
        units_per_reservoir (int): The number of units per reservoir.

        reservoir_amount (int, optional): The number of reservoirs. Defaults to 64.

        overlap (int, optional): The number of overlapping units between reservoir inputs. Defaults to 6.

        leak_rate (int, optional): The leak rate of the reservoirs. Defaults to 1.

        exponent (int, optional): The exponent of the PowerIndex layer (augmented hidden states). Defaults to 2.

        sigma (float, optional): The standard deviation of the input matrix. Defaults to 0.5.

        degree (int, optional): The degree of the Erdos-Renyi graph. Defaults to 2.

        spectral_radius (float, optional): The spectral radius of the reservoirs. Defaults to 0.99.

        seed (int, optional): The seed for the random number generator. Defaults to None, which means the seed is random.

        name (str, optional): The name of the model. Defaults to "Parallel_ESN".

        input_initializer (InputMatrix, optional): The initializer for the input matrix. Defaults to None, which means the default initializer is used.

        reservoir_initializer (ErdosRenyi, optional): The initializer for the reservoir matrix. Defaults to None, which means the default initializer is used.

        bias_initializer (keras.initializers.Zeros, optional): The initializer for the bias.
            Defaults to None, Zeros is used.
    """
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


######### READOUT FLAVOURS #########


def linear_readout(
    model,
    transient_data,
    train_data,
    train_target,
    output_dim=None,
    regularization=1e-8,
    method="ridge",
    solver="svd",  # This solver is the best
):
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
        model (keras.Model): The model with the linear readout.
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
    model = ModelWithReadout(model, readout_layer)

    # Have to build for some reason TODO
    model.build(transient_data.shape)

    # Calling the model in order to be able to save it TODO: Check if
    # this is necessary or better ways to do it
    model.predict(transient_data[:, :1, :], verbose=0)

    return model


def mlp_readout(
    model,
    transient_data,
    train_data,
    train_target,
    layers=1,
    internal_activation="relu",
    last_activation="linear",
    output_dim=None,
    epochs=100,
    learning_rate=0.001,
    plot_training=False,
    save_name=None,
):
    pass


def sgd_linear_readout(
    model,
    transient_data,
    train_data,
    train_target,
    learning_rate=0.001,
    epochs=200,
    regularization=1e-8,
):
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


####################################


#### Forecasting ####


def forecast(
    model,
    forecast_transient_data,
    val_data,
    val_target,
    forecast_length=100,
    callbacks=None,
    save_name=None,
):
    """Forecast the given data.

    Args:
        model (keras.Model): The model to be used for the forecast.

        forecast_transient_data (np.array): The data to be used to initialize the forecast.

        val_data (np.array): The data to be forecasted.

        val_target (np.array): The target data.

        forecast_length (int, optional): The length of the forecast. Defaults to 100.

        callbacks (list, optional): The list of callback functions
            to be used during the forecast. Defaults to None.

        save_name (str, optional): The path to save the forecast. Defaults to None.
            If None, the forecast is not saved.

    Returns:
        (np.array, float): The predictions and the loss.
    """
    forecast_length = min(forecast_length, val_data.shape[1])
    val_target = val_target[:, :forecast_length, :]

    print()
    print(
        f"Forecasting free running sequence {forecast_length} steps ahead.\n\n"
    )

    print("    Ensuring ESP...\n")
    print("    Forecast transient data shape: ", forecast_transient_data.shape)
    model.predict(forecast_transient_data)

    # Initialize predictions with the first element of the validation data
    predictions = val_data[:, :1, :]

    print()
    print("    Predicting...\n")

    # Initializing the monitored variables
    if callbacks is not None:
        monitored = {func.__name__: [] for func in callbacks}

    # Already tried initializing the predictions with shape (1, forecast_length, features) and the performance was similar
    for i in trange(forecast_length):
        pred = model(predictions[:, -1:, :])
        predictions = np.hstack((predictions, pred))

        # Accumulating the monitored variables
        if callbacks is not None:
            for func in callbacks:
                monitored[func.__name__].append(
                    func(pred[0], val_target[:, i, :])
                )

    # Eliminating the first element of the predictions
    predictions = predictions[:, 1:, :]
    print("    Predictions shape: ", predictions.shape)

    # Calculating the error
    try:
        loss = np.mean((predictions[0] - val_target[0]) ** 2)
    except ValueError:
        print("Error calculating the loss.")
        return np.inf

    print(f"Forecast loss: {loss}\n")

    if save_name is not None:
        print("Saving forecast...\n")
        # save the forecast in the forecasts folder
        np.save("".join([save_name, "_forecast.dt"]), predictions)

    if callbacks is not None:
        return predictions, monitored

    return predictions, None


def section_forecast(
    model,
    forecast_transient_data,
    val_data,
    val_target,
    section_length,
    section_initialization_length,
    number_of_sections=1,
    callbacks=None,
):
    """Forecast the given data in sections of length `section_length'.

    Everytime a section is forecasted, the model is reset back to zero and the last
    `section_initialization_length' elements corresponding to the val_target in the section
    are used to initialize the forecast of the next section.

    Args:
        model (keras.Model): The model to be used for the forecast.

        forecast_transient_data (np.array): The data to be used to initialize the whole forecast.

        val_data (np.array): The data to be forecasted.

        val_target (np.array): The target data.

        section_length (int): The length of each section.

        section_initialization_length (int): The length of the initialization of each section.

        number_of_sections (int, optional): The number of sections to be forecasted. Defaults to 1.

        callbacks (list, optional): The list of callback functions

    Returns:
        (np.array, dict): The predictions and the monitored variables.
    """

    # Initializing the monitored variables
    if callbacks is not None:
        monitored = {func.__name__: [] for func in callbacks}

    # Initializing the predictions
    predictions = np.zeros((1, 0, val_data.shape[2]))

    for i in range(number_of_sections):
        print(f"Forecasting section {i+1} of {number_of_sections}.\n")
        forecast_section, section_monitored = forecast(
            model,
            forecast_transient_data,
            val_data[:, i * section_length : (i + 1) * section_length, :],
            val_target[:, i * section_length : (i + 1) * section_length, :],
            forecast_length=section_length,
            callbacks=callbacks,
        )
        # Updating the predictions
        predictions = np.hstack((predictions, forecast_section))

        # Updating the forecast transient data
        forecast_transient_data = val_data[
            :,
            (i + 1) * section_length
            - section_initialization_length : (i + 1) * section_length,
            :,
        ]

        # Updating the monitored variables
        if callbacks is not None:
            for func in callbacks:
                monitored[func.__name__].append(
                    section_monitored[func.__name__]
                )

    if callbacks is not None:
        return predictions, monitored

    return predictions, None


#### Plotting functions ####


def plot_linear_forecast(
    predictions: np.ndarray,
    val_target: np.ndarray,
    dt=1,
    title="",
    save_path=None,
    show=True,
    ylabels=None,
    xlabel="t",
):
    """
    Plot the predictions and the target in a linear fashion.

    Args:
        predictions (np.array): The predictions of the model.

        val_target (np.array): The target values.

        dt (float): The time step between each prediction.

        title (str): The title of the plot.

        save_path (str): The path to save the plot.

        show (bool): Whether to show the plot or not.

        ylabels (list): The labels for each of the features.

        xlabel (str): The label for the x axis.

    Returns:
        None
    """
    forecast_length = predictions.shape[1]
    features = predictions.shape[-1]

    val_target = val_target[:, :forecast_length, :]

    xvalues = np.arange(0, forecast_length) * dt

    # Make each plot on a different axis
    fig, axs = plt.subplots(features, 1, sharey=True, figsize=(20, 9.6))
    fig.tight_layout()
    fig.subplots_adjust(top=0.9, bottom=0.08, hspace=0.3)

    fig.suptitle(title, fontsize=16)
    fig.supxlabel(xlabel)

    # Make this if-else better TODO
    if features == 1:
        axs.plot(xvalues, predictions[0, :, 0], label="prediction")
        axs.plot(xvalues, val_target[0, :, 0], label="target")
        if ylabels is not None:
            axs.set_ylabel(ylabels[0])
        axs.legend()
    else:
        for i in range(features):
            axs[i].plot(xvalues, predictions[0, :, i], label="prediction")
            axs[i].plot(xvalues, val_target[0, :, i], label="target")
            if ylabels is not None:
                axs[i].set_ylabel(ylabels[i])
            axs[i].legend()

    if save_path is not None:
        plt.savefig("".join([save_path, "_linear.png"]))

    if show:
        plt.show()


def plot_contourf_forecast(
    predictions,
    val_target,
    dt=1,
    title="",
    save_path=None,
    show=True,
    yvalues=None,
    xlabel=r"$\lambda_1 t$",
):
    """
    Plot the predictions and the target in a contourf fashion.

    Args:
        predictions (np.array): The predictions of the model.

        val_target (np.array): The target values.

        dt (float): The time step between each prediction.

        title (str): The title of the plot.

        save_path (str): The path to save the plot.

        show (bool): Whether to show the plot or not.

        yvalues (list): The values for each of the features.

        xlabel (str): The label for the x axis.

    Returns:
        None
    """
    forecast_length = predictions.shape[1]

    val_target = val_target[:, :forecast_length, :]

    xvalues = np.arange(0, forecast_length) * dt

    if yvalues is None:
        yvalues = np.arange(0, predictions.shape[-1])

    # Making the figure pretty
    fig, axs = plt.subplots(3, 1, sharey=True, figsize=[20, 9.6])
    fig.tight_layout()
    fig.subplots_adjust(top=0.9, bottom=0.08, right=1.1, hspace=0.3)

    fig.suptitle(title, fontsize=16)
    fig.supxlabel(xlabel)

    model_plot = axs[0].contourf(xvalues, yvalues, val_target[0].T, levels=50)
    axs[0].set_title("Original model")

    prediction_plot = axs[1].contourf(
        xvalues, yvalues, predictions[0].T, levels=50
    )
    axs[1].set_title("Predicted model")

    error = abs(predictions[0] - val_target[0])

    error_plot = axs[2].contourf(xvalues, yvalues, error.T, levels=20)
    axs[2].set_title("Error")

    # Individually adding the colorbars
    fig.colorbar(model_plot, ax=axs[0])
    fig.colorbar(prediction_plot, ax=axs[1])
    fig.colorbar(error_plot, ax=axs[2])

    # Common colorbar for all plots
    # fig.colorbar(model_plot, ax=axs.ravel().tolist(), shrink=0.5)

    if save_path is not None:
        plt.savefig("".join([save_path, "_contourf.png"]))

    if show:
        plt.show()


def render_video(
    data,
    predictions=None,
    title="",
    xlabel=r"N",
    frames=None,
    save_path=None,
    dt=1,
):
    """
    Render a video of the predictions and the target.

    Args:
        predictions (N x T x D np.array): The predictions of the model.

        val_target (N x T x D np.array) [Optional]: The target values. Real Data.

        title (str): The title of the plots.

        xlabel (str): The label for the x axis.

        frames (int): The number of frames to render. If None, render all of them.

        save_path (str): The path to save the plot. If None, the video is not saved.

    Returns:
        None
    """
    data = data.reshape(-1, data.shape[-1])

    if predictions is not None:
        # reshape the predictions and the target to 2D
        predictions = predictions.reshape(predictions.shape[1], -1)

    # Set the number of frames to the number of predictions
    if frames is None and predictions is not None:
        frames = predictions.shape[0]
    elif frames is None:
        frames = data.shape[0]

    fig, axs = plt.subplots(figsize=[20, 9.6])

    axs.set_ylim(data.min(), data.max())

    fig.tight_layout()
    fig.subplots_adjust(top=0.9, bottom=0.08)

    fig.suptitle(title, fontsize=16)
    fig.supxlabel(xlabel)

    (model_plot,) = axs.plot(data[0], label="Original model")

    # also write the time in the plot
    time_text = axs.text(0.02, 0.95, f"t={0}", transform=axs.transAxes)

    if predictions is not None:
        (prediction_plot,) = axs.plot(predictions[0], label="Predicted model")

    axs.legend()

    def update(frame):
        # if i % 10 == 0:
        #     print(f"Frame {i}/{frames}")

        # update the time in the plot
        time_text.set_text(f"t={frame*dt}")

        model_plot.set_ydata(data[frame])
        if predictions is not None:
            prediction_plot.set_ydata(predictions[frame])
            return model_plot, prediction_plot
        return (model_plot, time_text)

    print("Animating...")
    print()

    ani = FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=int(1000 * 2 * dt),
        blit=True,
        repeat=False,
    )

    # progress_callback function using tqdm to show the progress of the animation
    if save_path is not None:
        pbar = tqdm.tqdm(total=frames)

        def progress_callback(current_frame, total_frames):
            pbar.update(current_frame - pbar.n)

        ani.save(
            "".join([save_path, "_video.mp4"]),
            writer="ffmpeg",
            progress_callback=progress_callback,
        )

    plt.show()


#### Callbacks ####


def rms_error(predictions, target):
    """
    Calculate the RMS error between two 2D arrays.

    Args:
        predictions (np.array): The predictions of the model.

        target (np.array): The target values.

    Returns:
        (float): The RMS error.
    """
    return np.sqrt(np.mean((predictions - target) ** 2))


# Implement the plot of the relative maxima using argrelmax


def main():
    """Tryout many things."""
    init_transient = 1000
    transient = 1000
    train = 20000

    units = 4000
    spectral_radius = 0.45  # KS
    # spectral_radius = 1.21 # Mackey-Glass

    reservoir_amount = 16
    overlap = 6

    degree = 3  # KS
    # degree = 2  # Mackey-Glass
    # degree = 6  # Lorenz

    # sigma = 1
    sigma = 0.5  # KS
    # sigma = 0.1  # Lorenz

    forecast_len = 1000

    regularization = 1e-4  # KS
    # regularization = 1e-8  # Mackey-Glass/Lorenz

    L = 22  # KS
    N = 128  # KS

    max_lyap = lyap_ks(1, L)
    dt = 0.25 * max_lyap  # KS # delta t and max lyapunov time

    print("Max Lyapunov exponent: ", max_lyap)

    load_path = f"data/KS/L{L}_dt0.25/"

    name = f"KS_L{L}_N{N}_dt0.25_steps160000_diffusion-k1_run0.csv"
    # name = "mackey_alpha-0.2_beta-10_gamma-0.1_tau-17_n-150000.csv"
    # name = "Lorenz_[2, 2, 2]_tend3200_dt0.02.csv"

    title = f"Forecasting of the Kuramoto-Sivashinsky model with {units} units"  # KS
    # title = f"Forecasting of the Mackey-Glass model with {units} units"  # Mackey-Glass
    # title = f"Forecasting of the Lorenz model with {units} units"  # Lorenz

    save_name = (
        f"{name[:-4]}_units{units}_train{train}_inittransient{init_transient}"
        f"_transient{transient}_degree{degree}_sigma{sigma}_spectral_radius{spectral_radius}"
        f"_regularization{regularization}_forecast_len{forecast_len}"
    )

    (
        transient_data,
        train_data,
        train_target,
        forecast_transient_data,
        val_data,
        val_target,
    ) = load_data(
        load_path + name,
        transient=transient,
        train_length=train,
        init_transient=init_transient,
    )

    # ylabels = ["X", "Y", "z"]  # For the linear plot of lorenz
    yvalues = np.linspace(0, L, train_data.shape[-1])

    input_init = None
    bias_init = None

    # input_init = keras.initializers.RandomUniform(minval=-sigma, maxval=sigma)
    bias_init = keras.initializers.RandomUniform(minval=-sigma, maxval=sigma)
    reservoir_init = WattsStrogatzOwn(
        degree=degree,
        spectral_radius=spectral_radius,
        sigma=sigma,
        rewiring_p=0.5,
    )

    reservoir_init = RegularNX(
        degree=degree, spectral_radius=spectral_radius, sigma=sigma
    )

    # Simple ESN
    model = get_simple_esn(
        units=units,
        #   seed=seed,
        spectral_radius=spectral_radius,
        degree=degree,
        sigma=sigma,
        input_initializer=input_init,
        bias_initializer=bias_init,
        reservoir_initializer=reservoir_init,
        leak_rate=1,
    )

    # # Parallel ESN
    # model = get_parallel_esn(
    #     units_per_reservoir=units,
    #     reservoir_amount=reservoir_amount,
    #     overlap=overlap,
    #     #   seed=seed,
    #     spectral_radius=spectral_radius,
    #     degree=degree,
    #     sigma=sigma,
    #     input_initializer=input_init,
    #     bias_initializer=bias_init,
    #     reservoir_initializer=reservoir_init,
    # )

    # model.verify_esp(transient_data[:, :50, :], times=10)
    # exit(0)

    final_model = linear_readout(
        model,
        transient_data,
        train_data,
        train_target,
        regularization=regularization,
        method="ridge",
    )

    # final_model = sgd_linear_readout(
    #     model,
    #     transient_data,
    #     train_data,
    #     train_target,
    #     learning_rate=0.001,
    #     epochs=30,
    #     regularization=regularization,
    # )

    print(
        "Training finished, number of parameters: ", final_model.count_params()
    )

    keras.utils.plot_model(
        final_model.build_graph(),
        # show_shapes=True,
        to_file="parallel_model.png",
    )

    # final_model.save(
    #     "./Models/Parallel_ESN_KS_L22_N64_dt0.25_steps160000_diffusion-k1"
    # )
    # final_model = keras.models.load_model(
    #     "./Models/Parallel_ESN_KS_L22_N64_dt0.25_steps160000_diffusion-k1"
    # )
    # print("Model loaded")

    # exit(0)

    predictions, _ = forecast(
        final_model,
        forecast_transient_data,
        val_data,
        val_target,
        forecast_length=forecast_len,
    )

    # predictions, monitored = section_forecast(
    #     final_model,
    #     forecast_transient_data,
    #     val_data,
    #     val_target,
    #     section_initialization_length=50,
    #     section_length=100,
    #     number_of_sections=10,
    # )

    # plt.plot(monitored["rms_error"])
    # plt.show()

    # plot_linear_forecast(
    #     predictions,
    #     val_target,
    #     dt=dt,
    #     title=title,
    #     xlabel="t",
    #     ylabels=ylabels,
    #     save_path="plots/" + save_name,
    # )

    plot_contourf_forecast(
        predictions,
        val_target,
        dt=dt,
        title=title,
        save_path="plots/" + save_name,
        show=True,
        yvalues=yvalues,
    )

    print(val_target.shape)
    print(predictions.shape)

    # render_video(
    #     data=val_target,
    #     predictions=predictions,
    #     title=title,
    #     frames=forecast_len,
    #     save_path="videos/" + save_name + "_video" + ".mp4",
    #     dt=dt,
    # )


# # # # # # For hyperas


if __name__ == "__main__":
    print(timeit.timeit(main, number=1))


# help me organize this module in a better way
