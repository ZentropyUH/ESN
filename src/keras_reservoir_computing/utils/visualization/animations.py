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
    speed_factor: float = 1.0,
    savepath: Optional[str] = None,
    show: bool = True,
) -> FuncAnimation:
    """
    Animate 2D or 3D parametric trajectories with optional multiple samples,
    each displayed as a moving point with a trailing path.

    This function supports:
    - Single trajectory of shape (T, D)
    - Multiple trajectories of shape (N, T, D)

    Parameters
    ----------
    data : np.ndarray
        The trajectory data. Must be either:
         - 2D: (T, D), representing a single trajectory; or
         - 3D: (N, T, D), representing N separate trajectories.

        Here:
          - N is the number of samples (trajectories).
          - T is the number of time steps.
          - D is the spatial dimension (2 or 3).
    dt : float, optional
        The time step between frames. Takes precedence over ``fps`` if provided.
        Must be > 0 if specified.
    trail_length : float, optional
        Fraction of total frames used for the trailing path (default 0.1).
        E.g., if T=1000 and `trail_length`=0.1, the trail always covers 100 frames behind.
    sample_labels : list of str, optional
        A list of labels for each trajectory (length N). If None, uses "Sample i".
    separate_axes : bool, optional
        If True, each trajectory is plotted on a separate axis (organized in up to 3 columns).
        Otherwise, all trajectories appear in a single subplot.
    xlabels : str or list of str, optional
        x-axis label(s). If a single string, it’s reused for all subplots.
        If a list, must be length N.
    ylabels : str or list of str, optional
        y-axis label(s). Same behavior as ``xlabels``.
    zlabels : str or list of str, optional
        z-axis label(s). Only relevant if D=3. Same behavior as ``xlabels``.
    suptitle : str, optional
        A title for the entire figure (or each subplot if separate_axes).
    fps : int, optional
        Frames per second (default=60). Ignored if ``dt`` is provided.
    speed_factor : float, optional
        Scales the animation speed (default=1.0). For instance, speed_factor=2
        makes the animation twice as fast.
    savepath : str, optional
        If provided, saves the animation to the specified video file (requires ffmpeg).
    show : bool, optional
        Whether to display the animation in a window. Default is True.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        The resulting animation object.

    Raises
    ------
    ValueError
        If data shape is invalid, if ``D`` not in (2,3), or if the length of `sample_labels`
        doesn't match N, or if dt <= 0.

    Notes
    -----
    - In 2D, a grid is shown by default.
    - If multiple trajectories appear in the same axis, each has a different color/trail in the legend.
    - In 3D mode with separate_axes=True, the camera angle is synchronized across subplots.
    - The argument ``trail_length`` is converted into an integer number of frames behind
      the current position.
    - Setting ``show=False`` still creates the animation; it just won't pop up a figure window.
    """
    # Basic shape checks.
    ndim = data.ndim
    if ndim == 2:
        # Single sample: shape (T, D)
        data = data[np.newaxis, ...]  # Make it (1, T, D)
    elif ndim != 3:
        raise ValueError(f"Data must be 2D (T, D) or 3D (N, T, D). Received shape {data.shape}.")

    N, T, D = data.shape
    if D not in (2, 3):
        raise ValueError(f"Data must have D=2 or 3, but got D={D}.")

    if dt is not None:
        if dt <= 0:
            raise ValueError(f"dt must be positive, got dt={dt}.")

    # Prepare default labels.
    sample_labels = sample_labels or [f"Sample {i+1}" for i in range(N)]
    if len(sample_labels) != N:
        raise ValueError(
            f"Length of sample_labels ({len(sample_labels)}) must match number of samples ({N})."
        )

    # Expand x/y/z labels to match N if they are given as a single string.
    if isinstance(xlabels, str):
        xlabels = [xlabels] * N
    elif len(xlabels) != N:
        raise ValueError(f"Expected {N} x-labels, got {len(xlabels)}.")

    if isinstance(ylabels, str):
        ylabels = [ylabels] * N
    elif len(ylabels) != N:
        raise ValueError(f"Expected {N} y-labels, got {len(ylabels)}.")

    if D == 3:
        if isinstance(zlabels, str):
            zlabels = [zlabels] * N
        elif len(zlabels) != N:
            raise ValueError(f"Expected {N} z-labels, got {len(zlabels)}.")

    # Convert trail_length fraction to number of frames
    trail_frames = max(1, int(T * trail_length))

    # Compute animation interval (milliseconds)
    if dt is not None:
        interval = 1000 * dt / speed_factor
        # Recompute fps from dt so we can use it for saving
        fps = (1.0 / dt) * speed_factor
    else:
        interval = (1000.0 / fps) * speed_factor
        fps *= speed_factor

    is_3d = D == 3

    # Figure and axes
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
        # Remove extra axes if N < rows*cols
        for i in range(N, len(axes)):
            fig.delaxes(axes[i])
        if suptitle:
            fig.suptitle(suptitle, fontsize=14)
            fig.subplots_adjust(top=0.85)
    else:
        fig = plt.figure(figsize=(7, 7))
        if is_3d:
            ax = fig.add_subplot(111, projection="3d")
        else:
            ax = fig.add_subplot(111)
            ax.grid(True)
        axes = [ax] * N

        if suptitle:
            fig.suptitle(suptitle, fontsize=14, y=0.95)
            fig.subplots_adjust(top=0.80)

    # Initialize containers for point markers and trailing lines
    points = []
    trails = []

    # Set up each sample’s line
    for i in range(N):
        ax_i = axes[i] if separate_axes else axes[0]
        x = data[i, :, 0]
        y = data[i, :, 1]
        z = data[i, :, 2] if is_3d else None

        # Axes limits
        margin = 1.0
        ax_i.set_xlim(x.min() - margin, x.max() + margin)
        ax_i.set_ylim(y.min() - margin, y.max() + margin)
        if is_3d and z is not None:
            ax_i.set_zlim(z.min() - margin, z.max() + margin)

        # Axis labels
        ax_i.set_xlabel(xlabels[i])
        ax_i.set_ylabel(ylabels[i])
        if is_3d:
            ax_i.set_zlabel(zlabels[i])

        if separate_axes and N > 1:
            ax_i.set_title(sample_labels[i])

        # Create point markers and trailing lines
        if is_3d:
            (point_line,) = ax_i.plot([], [], [], "o", markersize=5)
            (trail_line,) = ax_i.plot(
                [], [], [], "-", label=sample_labels[i], alpha=0.7, linewidth=2
            )
        else:
            (point_line,) = ax_i.plot([], [], "o", markersize=5)
            (trail_line,) = ax_i.plot([], [], "-", label=sample_labels[i], alpha=0.7, linewidth=2)

        points.append(point_line)
        trails.append(trail_line)

    # If multiple samples are in one subplot, show legend
    if not separate_axes and N > 1:
        axes[0].legend(loc="upper right")

    def init():
        for point_line, trail_line in zip(points, trails):
            if is_3d:
                point_line.set_data([], [])
                point_line.set_3d_properties([])
                trail_line.set_data([], [])
                trail_line.set_3d_properties([])
            else:
                point_line.set_data([], [])
                trail_line.set_data([], [])
        return points + trails

    def update(frame):
        for i in range(N):
            x = data[i, :, 0]
            y = data[i, :, 1]
            z = data[i, :, 2] if is_3d else None
            start = max(0, frame - trail_frames)

            if is_3d and z is not None:
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
        repeat=show,  # If show=False, do not repeat
    )

    # Synchronize 3D camera angles across subplots
    if separate_axes and is_3d:

        def on_move(event):
            if event.inaxes not in axes:
                return
            elev, azim = event.inaxes.elev, event.inaxes.azim
            for ax_ in axes:
                if (ax_.elev, ax_.azim) != (elev, azim):
                    ax_.view_init(elev=elev, azim=azim)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", on_move)

    if show:
        plt.show()

    # Optionally save the animation via ffmpeg
    if savepath:
        with Progress() as progress:
            task = progress.add_task("[cyan]Saving animation...", total=T)

            def progress_callback(i, n):
                progress.update(task, completed=i + 1)

            writer = FFMpegWriter(
                fps=fps,
                extra_args=["-threads", "0"],  # Let ffmpeg decide thread count
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
    lam: float = 1.0,
    sample_labels: Optional[List[str]] = None,
    separate_axes: bool = False,
    xlabels: Union[str, List[str]] = "X",
    ylabels: Union[str, List[str]] = "Y",
    suptitle: Optional[str] = None,
    fps: int = 60,
    speed_factor: float = 1.0,
    savepath: Optional[str] = None,
    show: bool = True,
) -> FuncAnimation:
    """
    Animate 2D "time series" data where each frame shows the feature values at a given time,
    plotted as a 1D slice on the y-axis (feature index on the x-axis).

    Parameters
    ----------
    data : np.ndarray
        Array of shape (T, D) or (N, T, D), where:
         - T is the number of time steps (frames).
         - D is the number of features at each time step.
         - N is the number of separate samples (optional).
    dt : float, optional
        Time step between frames (seconds or arbitrary units). If provided (>0),
        overrides ``fps`` for controlling the animation speed.
    lam : float, optional
        Scaling factor for the "time" displayed, e.g., normalizing by a Lyapunov exponent.
        By default, 1.0 (no scaling).
    sample_labels : list of str, optional
        Labels for each sample (must match N if multiple samples).
    separate_axes : bool, optional
        If True, each sample is plotted on its own subplot. Otherwise, all samples
        appear in a single axis, distinguished by color/label.
    xlabels : str or list of str, optional
        Label(s) for the x-axis.
    ylabels : str or list of str, optional
        Label(s) for the y-axis.
    suptitle : str, optional
        Title for the figure.
    fps : int, optional
        Frames per second if dt is not specified (default=60).
    speed_factor : float, optional
        Factor controlling animation speed (default=1.0).
    savepath : str, optional
        If provided, saves the animation via ffmpeg to the specified file.
    show : bool, optional
        If True (default), displays the animation window.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        The resulting animation object.

    Raises
    ------
    ValueError
        If `data` doesn't match the required shape, or if dt <= 0.

    Notes
    -----
    - Each frame corresponds to a single time step from 0..T-1.
    - The x-axis (0..D) denotes the feature indices; the y-axis is the magnitude of features.
    - The time text in the corner is scaled by `dt * lam`. If `dt` is not provided, time
      is displayed as frame index.
    - If multiple samples share a single axis, a legend distinguishes them.
    - Setting ``repeat=show`` ensures that if `show=False`, the animation won't loop.
    """
    ndim = data.ndim
    if ndim == 2:
        # Single sample: (T, D)
        data = data[np.newaxis, ...]  # (1, T, D)
    elif ndim != 3:
        raise ValueError(f"Data must be 2D (T, D) or 3D (N, T, D). Got shape {data.shape}.")

    N, T, D = data.shape

    if dt is not None:
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}.")

    # Default labels for samples
    sample_labels = sample_labels or [f"Sample {i+1}" for i in range(N)]
    if len(sample_labels) != N:
        raise ValueError(f"Length of sample_labels ({len(sample_labels)}) must match N={N}.")

    # Expand x/y labels
    if isinstance(xlabels, str):
        xlabels = [xlabels] * N
    elif len(xlabels) != N:
        raise ValueError(f"Expected {N} x-labels, got {len(xlabels)}.")

    if isinstance(ylabels, str):
        ylabels = [ylabels] * N
    elif len(ylabels) != N:
        raise ValueError(f"Expected {N} y-labels, got {len(ylabels)}.")

    # Compute animation interval
    if dt is not None:
        interval = (1000.0 * dt) / speed_factor
        fps = (1.0 / dt) * speed_factor
    else:
        interval = (1000.0 / fps) * speed_factor
        fps *= speed_factor

    # Create figure & axes
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

        # Remove extra axes if needed
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

    # Prepare lines
    lines = []
    # y-limits for all data
    y_min, y_max = data.min(), data.max()

    for i in range(N):
        ax_i = axes[i] if separate_axes else axes[0]

        ax_i.set_xlim(0, D - 1)
        ax_i.set_ylim(y_min, y_max)
        ax_i.set_xlabel(xlabels[i])
        ax_i.set_ylabel(ylabels[i])

        if separate_axes and N > 1:
            ax_i.set_title(sample_labels[i])

        (line,) = ax_i.plot([], [], label=sample_labels[i])
        lines.append(line)

    if not separate_axes and N > 1:
        axes[0].legend(loc="upper right")

    # Add a time text box
    time_text = fig.text(0.85, 0.90, "", fontsize=9, ha="right", va="top")

    def init():
        for line in lines:
            line.set_data([], [])
        time_text.set_text("")
        return lines + [time_text]

    def update(frame):
        # For each sample i, we plot the D feature values at time=frame
        x_vals = np.arange(D)
        for i in range(N):
            y_vals = data[i, frame, :]
            lines[i].set_data(x_vals, y_vals)

        if dt is not None:
            current_time = frame * dt * lam
            time_text.set_text(rf"$t={current_time:.2f}$")
        else:
            time_text.set_text(rf"Frame={frame:d}")

        return lines + [time_text]

    ani = FuncAnimation(
        fig=fig,
        func=update,
        frames=T,
        init_func=init,
        interval=interval,
        repeat=show,  # match animate_trail's logic
    )

    if show:
        plt.show()

    if savepath:
        with Progress() as progress:
            task = progress.add_task("[cyan]Saving animation...", total=T)

            def progress_callback(i, n):
                progress.update(task, completed=i + 1)

            writer = FFMpegWriter(
                fps=fps,
                extra_args=["-threads", "0"],
            )
            ani.save(filename=savepath, writer=writer, progress_callback=progress_callback)

    return ani


__all__ = ["animate_trail", "animate_timeseries"]


def __dir__():
    return __all__
