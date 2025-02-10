from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from rich.progress import Progress


def animate_trail(
    data: np.ndarray,
    dt: Optional[float] = None,
    trail_length: float = 0.1,
    sample_labels: Optional[List[str]] = None,
    separate_axes: bool = False,
    xlabels: Union[str, List[str]] = "X",
    ylabels: Union[str, List[str]] = "Y",
    zlabels: Union[str, List[str]] = "Z",
    suptitle: Optional[str] = None,
    fps: int = 60,
    speed_factor: float = 1,
    savepath: Optional[str] = None,
    show: bool = True,
) -> FuncAnimation:
    """
    Animates 2D or 3D parametric trajectories with optional multiple samples.
    Each sample is visualized as a moving point with a trailing path.

    The function supports both single and multiple trajectories:
    - If `data` has shape (N, T, D), it contains N separate trajectories.
    - If `data` has shape (T, D), it is treated as a single trajectory (N=1).
    - 2D data (D=2) is plotted in 2D, while 3D data (D=3) is plotted in 3D.

    Parameters
    ----------
    data : np.ndarray
        - Shape (N, T, D) or (T, D).
        - N: Number of separate trajectories (optional).
        - T: Number of time steps.
        - D: Dimensionality (2 for 2D, 3 for 3D).
    dt : float, optional
        Time step between frames. Used to determine animation speed if provided.
    trail_length : float, optional
        Length of the trailing path as a fraction of total frames (default=0.1).
    sample_labels : list of str, optional
        Labels for each trajectory (length N).
    separate_axes : bool, optional
        If True, each trajectory is plotted in its own subplot (max 3 per row).
        Otherwise, all are plotted in a single subplot.
    xlabels : str or list of str, optional
        Label(s) for the x-axis. Can be a single string (applied to all) or a list of length N.
    ylabels : str or list of str, optional
        Label(s) for the y-axis.
    zlabels : str or list of str, optional
        Label(s) for the z-axis (only for 3D plots).
    suptitle : str, optional
        Title for the entire figure or each subplot (if `separate_axes=True`).
    fps : int, optional
        Frames per second for the animation (default=60). Ignored if `dt` is provided.
    speed_factor : float, optional
        Factor to control animation speed (default=1).
    savepath : str, optional
        If provided, saves the animation as a video file (requires `ffmpeg`).
    show : bool, optional
        Whether to display the animation.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        The generated animation object.

    Notes
    -----
    - In 2D mode, a grid is shown by default.
    - If `separate_axes` is True, subplots are arranged in rows of 3.
    - If multiple trajectories are plotted in a single subplot, the legend corresponds to the trail colors.
    - If `dt` is provided, it takes precedence over `fps` for controlling speed.
    - In 3D mode with `separate_axes=True`, all plots views are synchronized in real-time.
    """
    # Ensure data is (N, T, D).
    ndim = data.ndim
    if ndim == 2:
        data = data[np.newaxis, ...]  # Add dimension for single sample
    elif ndim > 3 or ndim == 1:
        raise ValueError(
            f"Data must be either 2D (T, D) or 3D (N, T, D), (got {data.shape})."
        )

    N, T, D = data.shape

    if D not in (2, 3):
        raise ValueError(
            f"Data must have D=2 for 2D or D=3 for 3D animation, got D={D}."
        )

    is_3d = D == 3

    # Prepare labels
    sample_labels = sample_labels or [f"Sample {i+1}" for i in range(N)]
    if len(sample_labels) != N:
        raise ValueError(
            f"Length of sample_labels must match the number of samples {N}, got {len(sample_labels)}."
        )

    if isinstance(xlabels, str):
        xlabels = [xlabels] * N
    elif len(xlabels) != N:
        raise ValueError(f"Expected {N} x-labels, got {len(xlabels)}.")

    if isinstance(ylabels, str):
        ylabels = [ylabels] * N
    elif len(ylabels) != N:
        raise ValueError(f"Expected {N} y-labels, got {len(ylabels)}.")

    if is_3d:
        if isinstance(zlabels, str):
            zlabels = [zlabels] * N
        elif len(zlabels) != N:
            raise ValueError(f"Expected {N} z-labels, got {len(zlabels)}.")

    # Convert trail_length from fraction to absolute number of frames.
    trail_frames = max(1, int(T * trail_length))

    # Compute animation interval (ms) based on dt or fps.
    if dt is not None:
        interval = (1000 * dt) / speed_factor
        fps = (1 / dt) * speed_factor
    else:
        interval = (1000 / fps) * speed_factor
        fps = fps * speed_factor

    # Create figure and axes.
    if separate_axes:
        cols = min(N, 3)
        rows = int(np.ceil(N / cols))
        fig, axes = plt.subplots(
            nrows=rows,
            ncols=cols,
            figsize=(5 * cols, 5 * rows),
            subplot_kw={"projection": "3d"} if is_3d else {},
        )
        axes = np.ravel(axes)
        for i in range(N, len(axes)):
            fig.delaxes(axes[i])
        if suptitle:
            fig.suptitle(suptitle, fontsize=14)
            fig.subplots_adjust(top=0.85)

    else:
        fig = plt.figure(figsize=(7, 7))

        # Decide whether to use 3D or 2D plot.
        # If 2D, add grid.
        if is_3d:
            ax = fig.add_subplot(111, projection="3d")
        else:
            ax = fig.add_subplot(111)
            ax.grid(True)

        axes = [ax] * N

        if suptitle:
            fig.suptitle(suptitle, fontsize=14, y=0.95)
            fig.subplots_adjust(top=0.80)

    # Containers for point and trail lines.
    points = []
    trails = []

    # Set up each sample.
    for i in range(N):
        ax_i = axes[i] if separate_axes else axes[0]
        x = data[i, :, 0]
        y = data[i, :, 1]
        z = data[i, :, 2] if is_3d else None

        ax_i.set_xlim(min(x) - 1, max(x) + 1)
        ax_i.set_ylim(min(y) - 1, max(y) + 1)

        if is_3d:
            ax_i.set_zlim(min(z) - 1, max(z) + 1)
        else:
            ax_i.grid(True)

        ax_i.set_xlabel(xlabels[i])
        ax_i.set_ylabel(ylabels[i])
        if is_3d:
            ax_i.set_zlabel(zlabels[i])

        if separate_axes and N > 1:
            ax_i.set_title(sample_labels[i])

        # Create point and trail. Assign label to the trail so that the legend shows the trail color.
        if is_3d:
            (point_line,) = ax_i.plot([], [], [], "o", markersize=5)
            (trail_line,) = ax_i.plot(
                [], [], [], "-", label=sample_labels[i], alpha=0.7, linewidth=2
            )
        else:
            (point_line,) = ax_i.plot([], [], "o", markersize=5)
            (trail_line,) = ax_i.plot(
                [], [], "-", label=sample_labels[i], alpha=0.7, linewidth=2
            )

        points.append(point_line)
        trails.append(trail_line)

    if not separate_axes and N > 1:
        axes[0].legend(loc="upper right")

    def init():
        for point_line, trail_line in zip(points, trails):
            point_line.set_data([], [])
            trail_line.set_data([], [])
            if is_3d:
                point_line.set_3d_properties([])
                trail_line.set_3d_properties([])
        return points + trails

    def update(frame):
        for i in range(N):
            x = data[i, :, 0]
            y = data[i, :, 1]
            z = data[i, :, 2] if is_3d else None
            start = max(0, frame - trail_frames)

            if is_3d:
                points[i].set_data([x[frame]], [y[frame]])
                points[i].set_3d_properties([z[frame]])
                trails[i].set_data(x[start : frame + 1], y[start : frame + 1])
                trails[i].set_3d_properties(z[start : frame + 1])
            else:
                points[i].set_data([x[frame]], [y[frame]])
                trails[i].set_data(x[start : frame + 1], y[start : frame + 1])
        return points + trails

    ani = FuncAnimation(
        fig=fig,
        func=update,
        frames=T,
        init_func=init,
        blit=True,
        interval=interval,
        repeat=show,
    )

    # Synchronize 3D views in real time when separate_axes=True.
    if separate_axes and is_3d:

        def on_move(event):
            if event.inaxes not in axes:
                return
            elev, azim = event.inaxes.elev, event.inaxes.azim
            for ax in axes:
                if (ax.elev, ax.azim) != (elev, azim):
                    ax.view_init(elev=elev, azim=azim)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", on_move)

    if show:
        plt.show()

    if savepath:
        with Progress() as progress:
            task = progress.add_task(
                "[cyan]Saving animation...", total=T
            )  # Set total frames

            def progress_callback(i, n):
                progress.update(task, completed=i + 1)  # Update progress bar

            writer = FFMpegWriter(
                fps=fps,
                extra_args=[
                    "-threads",
                    "0",
                ],  # Use multiple threads. Let ffmpeg decide the ammount.
            )

            ani.save(
                filename=savepath,
                writer=writer,
                progress_callback=progress_callback,
            )

    return ani


def animate_timeseries(
    data: np.ndarray,
    dt: Optional[float] = None,
    lam: Optional[float] = 1,
    sample_labels: Optional[List[str]] = None,
    separate_axes: bool = False,
    xlabels: Union[str, List[str]] = "X",
    ylabels: Union[str, List[str]] = "Y",
    suptitle: Optional[str] = None,
    fps: int = 60,
    speed_factor: float = 1.0,
    savepath: Optional[str] = None,
    show: bool = True,
):
    """
    Animates 2D time series data where each sample is displayed as a dynamically updating line plot.

    Parameters
    ----------
    data : np.ndarray
        - Shape (N, T, D) or (T, D).
        - N: Number of samples (optional).
        - T: Number of time steps.
        - D: Number of features.
    dt : float, optional
        Time step between frames. Used for animation timing.
    lam : float, optional
        Scaling factor for the time axis (e.g., normalization by a Lyapunov exponent).
    sample_labels : list of str, optional
        Labels for each sample (length N).
    separate_axes : bool, optional
        If True, each sample is plotted in a separate subplot (max 3 per row).
        Otherwise, all samples are displayed on the same axis.
    xlabels : str or list of str, optional
        Label(s) for the x-axis.
    ylabels : str or list of str, optional
        Label(s) for the y-axis.
    suptitle : str, optional
        Title for the figure.
    fps : int, optional
        Frames per second (default=60). Ignored if `dt` is provided.
    speed_factor : float, optional
        Factor to control animation speed (default=1.0).
    savepath : str, optional
        If provided, saves the animation as a video file.
    show : bool, optional
        Whether to display the animation.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        The generated animation object.

    Notes
    -----
    - The time axis is automatically scaled using `dt * lam`.
    - If `separate_axes` is False and multiple samples are plotted together, the legend identifies different samples.
    """
    # Ensure data is at least 2D
    ndim = data.ndim
    if ndim == 2:
        # data shape: (T, D)
        data = data[np.newaxis, ...]  # Make it (1, T, D)
    elif ndim > 3 or ndim == 1:
        raise ValueError(
            f"Data must be either 2D (T, D) or 3D (N, T, D), got {data.shape}."
        )

    N, T, D = data.shape

    # Prepare labels
    sample_labels = sample_labels or [f"Sample {i}" for i in range(N)]
    if len(sample_labels) != N:
        raise ValueError(
            f"Length of sample_labels must match the number of samples {N}, got {len(sample_labels)}."
        )

    if isinstance(xlabels, str):
        xlabels = [xlabels] * N
    elif len(xlabels) != N:
        raise ValueError(f"Expected {N} x-labels, got {len(xlabels)}.")

    if isinstance(ylabels, str):
        ylabels = [ylabels] * N
    elif len(ylabels) != N:
        raise ValueError(f"Expected {N} y-labels, got {len(ylabels)}.")

    # Compute animation interval (ms) based on dt or fps
    if dt is not None:
        interval = (1000 * dt) / speed_factor
        fps = (1 / dt) * speed_factor
    else:
        interval = (1000 / fps) * speed_factor
        fps = fps * speed_factor

    # Create figure and axes
    if separate_axes and N > 1:
        cols = min(N, 3)
        rows = int(np.ceil(N / cols))
        fig, axes = plt.subplots(
            nrows=rows,
            ncols=cols,
            figsize=(5 * cols, 5 * rows),
            sharex=True,
            sharey=False,
        )

        axes = np.ravel(axes)

        for i in range(N, len(axes)):
            fig.delaxes(axes[i])
        if suptitle:
            fig.suptitle(suptitle, fontsize=14)
            fig.subplots_adjust(top=0.85)

    else:
        fig, ax = plt.subplots(figsize=(7, 7))
        axes = [ax] * N
        if suptitle:
            fig.suptitle(suptitle, fontsize=14, y=0.95)
            fig.subplots_adjust(top=0.80)

    # Prepare lines (and possibly legends)
    lines = []
    for i in range(N):
        ax_i = axes[i] if separate_axes else axes[0]

        ax_i.set_xlim(0, D)
        ax_i.set_ylim(np.min(data), np.max(data))
        ax_i.set_xlabel(xlabels[i])
        ax_i.set_ylabel(ylabels[i])

        if separate_axes and N > 1:
            ax_i.set_title(sample_labels[i])

        (line,) = ax_i.plot([], [], label=sample_labels[i])

        lines.append(line)

    if not separate_axes and N > 1:
        axes[0].legend(loc="upper right")

    # Time text
    time_text = fig.text(0.85, 0.90, "", fontsize=9, ha="right", va="top")

    # Initialization
    def init():
        for line in lines:
            line.set_data([], [])
        time_text.set_text("")
        return lines + [time_text]

    # Animation update
    def update(frame):
        for i in range(N):
            x_vals = np.arange(D)
            y_vals = data[i, frame, :]
            lines[i].set_data(x_vals, y_vals)

        if dt is not None:
            current_time = frame * dt * lam
            time_text.set_text(rf"$t/T_\lambda: {current_time:.2f}$")
        else:
            current_time = frame
            time_text.set_text(rf"$T: {current_time:.2f}$")

        return lines + [time_text]

    ani = FuncAnimation(
        fig=fig,
        func=update,
        frames=T,
        init_func=init,
        interval=interval,
        repeat=show,
    )

    if show:
        plt.show()

    # Save if requested
    if savepath:
        with Progress() as progress:
            task = progress.add_task(
                "[cyan]Saving animation...", total=T
            )  # Set total frames

            def progress_callback(i, n):
                progress.update(task, completed=i + 1)  # Update progress bar

            writer = FFMpegWriter(
                fps=fps,
                extra_args=[
                    "-threads",
                    "0",
                ],  # Use multiple threads. Let ffmpeg decide the ammount.
            )

            ani.save(
                filename=savepath,
                writer=writer,
                progress_callback=progress_callback,
            )

    return ani


__all__ = ["animate_trail", "animate_timeseries"]

def __dir__():
    return __all__