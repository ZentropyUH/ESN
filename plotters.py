"""Define the plotting functions for the data. This includes the plotting of the predictions and the target data."""
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from matplotlib.animation import FuncAnimation

# TODO: Change the plotting functions to read the predictions and the target from files.

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


def plot_cumulative_rmse(
    predictions,
    val_target,
    dt=1,
    title="",
    save_path=None,
    show=True,
    ylabels=None,
    xlabel="t",
):
    """
    Plot the cumulative RMSE of the predictions and the target.

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
        axs.plot(xvalues, np.cumsum(predictions[0, :, 0]), label="prediction")
        axs.plot(xvalues, np.cumsum(val_target[0, :, 0]), label="target")
        if ylabels is not None:
            axs.set_ylabel(ylabels[0])
        axs.legend()
    else:
        for i in range(features):
            axs[i].plot(
                xvalues, np.cumsum(predictions[0, :, i]), label="prediction"
            )
            axs[i].plot(
                xvalues, np.cumsum(val_target[0, :, i]), label="target"
            )
            if ylabels is not None:
                axs[i].set_ylabel(ylabels[i])
            axs[i].legend()

    if save_path is not None:
        plt.savefig("".join([save_path, "_cumulative_rmse.png"]))

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

        # pylint: disable=unused-argument
        def progress_callback(current_frame, total_frames):
            pbar.update(current_frame - pbar.n)

        ani.save(
            "".join([save_path, "_video.mp4"]),
            writer="ffmpeg",
            progress_callback=progress_callback,
        )

    plt.show()
