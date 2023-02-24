"""Define some general utility functions."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from matplotlib.animation import FuncAnimation
from tqdm import trange

#### Parameters ####


def lyap_ks(i, l):
    """Estimation of the i-th largest Lyapunov Time of the KS model.

    Taken from the paper:
        "Lyapunov Exponents of the Kuramoto-Sivashinsky PDE. arxiv:1902.09651v1"
    """
    # This approximation is taken from the above paper. Verify veracity.
    return 0.093 - 0.94 * (i - 0.39) / l


def load_data(
    name: str,
    transient: int = 1000,
    train_length: int = 5000,
    init_transient: int = 0,
    step: int = 1,
) -> tuple:
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

    print("Original data shape: ", data.shape)
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

    print(f"data shape: {data.shape}")

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

    # Already tried initializing the predictions with shape (1, forecast_length, features)
    # and the performance was similar
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
    Calculate the cumulative RMS error between two 2D arrays.

    Args:
        predictions (np.array): The predictions of the model. Array of shape (T, D)

        target (np.array): The target values. Array of shape (T, D)

    Returns:
        (float): The RMS error.
    """

    return np.sqrt(np.mean((predictions - target) ** 2))


# Implement the plot of the relative maxima using argrelmax
