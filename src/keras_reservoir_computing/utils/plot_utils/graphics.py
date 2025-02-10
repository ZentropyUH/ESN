from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np


def plot_2d_timeseries(
    data: np.ndarray,
    dt: float = 1.0,
    lam: float = 1.0,
    suptitle: Optional[str] = None,
    sample_labels: Optional[List[str]] = None,
    feature_labels: Optional[List[str]] = None,
    xlabels: Union[str, List[str]] = "Time",
    ylabels: Union[str, List[str]] = "Feature Value",
    separate_axes: bool = False,
    yscale: str = "linear",
    xscale: str = "linear",
    **kwargs,
):
    """
    Plots 2D time series data with optional multiple samples and features.

    Parameters
    ----------
    data : np.ndarray
        Input array of shape (N, T, D) or (T, D).
        - N: Number of samples (optional).
        - T: Number of time steps.
        - D: Number of features.
    dt : float
        Time step between consecutive data points.
    lam : float
        Scaling factor for the time axis.
    suptitle : str, optional
        Title for the entire figure.
    sample_labels : list of str, optional
        Labels for each sample.
    feature_labels : list of str, optional
        Labels for each feature.
    xlabel : str or list of str
        Label(s) for the x-axis.
    ylabel : str or list of str
        Label(s) for the y-axis.
    separate_axes : bool
        If True, each feature is plotted on a separate subplot.
    yscale : {'linear', 'log'}
        Scale for the y-axis.
    xscale : {'linear', 'log'}
        Scale for the x-axis.
    **kwargs : dict
        Additional keyword arguments passed to `plt.plot()`.

    Returns
    -------
    list of matplotlib.axes.Axes
        Axes objects corresponding to the plots.

    Notes
    -----
    - If `separate_axes` is True, each feature is plotted individually.
    - If multiple samples exist and in a single plot, legends group samples in a structured manner.
    """
    if data.ndim > 3:
        raise ValueError(f"Data must have ndim <= 3 (got {data.ndim}).")
    if data.ndim == 1:
        data = data[np.newaxis, ..., np.newaxis]
    if data.ndim == 2:
        data = data[np.newaxis, ...]

    N, T, D = data.shape

    t = np.arange(T) * dt * lam

    fig, axes = (
        plt.subplots(D, 1, figsize=(12, 2 * D), sharex=True)
        if separate_axes
        else (plt.figure(figsize=(12, 3)), plt.gca())
    )

    axes = np.atleast_1d(axes)  # Ensure iterable structure for consistency

    if suptitle:
        fig.suptitle(suptitle, fontsize=14)

    # Generate default labels if not provided
    sample_labels = sample_labels or [f"Sample {i+1}" for i in range(N)]
    feature_labels = feature_labels or [f"Feature {j+1}" for j in range(D)]

    if len(sample_labels) != N:
        raise ValueError(f"Length of sample_labels must be {N}.")
    if len(feature_labels) != D:
        raise ValueError(f"Length of feature_labels must be {D}.")

    # Handle axis labels
    # Ensure xlabel and ylabel are lists of length D
    if isinstance(xlabels, str):
        xlabels = [xlabels] * D
    elif len(xlabels) != D:
        raise ValueError(f"Expected {D} x-labels, got {len(xlabels)}.")

    if isinstance(ylabels, str):
        ylabels = [ylabels] * D
    elif len(ylabels) != D:
        raise ValueError(f"Expected {D} y-labels, got {len(ylabels)}.")

    # Plot data
    for feature_idx, ax in enumerate(axes):
        for sample_idx in range(N):
            # Label explanation:
            # "Sample 1 - Feature 1" if N > 1 and not separate_axes,
            # else "Sample 1" if separate_axes,
            # else "Feature 1" if N = 1.
            label = (
                f"{sample_labels[sample_idx]} - {feature_labels[feature_idx]}"
                if N > 1 and not separate_axes
                else (
                    sample_labels[sample_idx]
                    if separate_axes and N > 1
                    else feature_labels[feature_idx]
                )
            )
            ax.plot(t, data[sample_idx, :, feature_idx], label=label, **kwargs)

        ax.set_ylabel(ylabels[feature_idx])
        ax.set_xlabel(xlabels[feature_idx])
        ax.set_yscale(yscale)
        ax.set_xscale(xscale)
        ax.set_title(feature_labels[feature_idx])

    # Manage legends
    if separate_axes:
        for ax in axes:
            ax.legend(loc="upper left")
        plt.tight_layout()
    else:
        handles, legend_labels = axes[0].get_legend_handles_labels()
        if N == 1:
            axes[0].legend(handles, legend_labels, loc="upper left", frameon=True)
        else:
            grouped_handles = {sample: [] for sample in sample_labels}
            grouped_labels = {sample: [] for sample in sample_labels}

            for h, lbl in zip(handles, legend_labels):
                for sample in sample_labels:
                    if lbl.startswith(sample):
                        grouped_handles[sample].append(h)
                        grouped_labels[sample].append(lbl)
                        break

            ordered_handles = sum(grouped_handles.values(), [])
            ordered_labels = sum(grouped_labels.values(), [])

            axes[0].legend(
                ordered_handles,
                ordered_labels,
                ncol=N,
                title="Grouped by Samples",
                loc="upper left",
                frameon=True,
            )

    return axes


def plot_3d_parametric(
    data: np.ndarray,
    suptitle: Optional[str] = None,
    sample_labels: Optional[List[str]] = None,
    separate_axes: bool = False,
    xlabels: Union[str, List[str]] = "X",
    ylabels: Union[str, List[str]] = "Y",
    zlabels: Union[str, List[str]] = "Z",
    **kwargs,
):
    """
    Plots 3D parametric curves from trajectory data.

    Each trajectory represents a parametric function x(t), y(t), z(t).
    Supports multiple samples and the option to use separate subplots.

    Parameters
    ----------
    data : np.ndarray
        Input array of shape (N, T, 3) or (T, 3).
        - N: Number of trajectories (optional).
        - T: Number of time steps.
    suptitle : str, optional
        Title for the figure.
    sample_labels : list of str, optional
        Labels for each trajectory.
    separate_axes : bool, optional
        If True, each trajectory is plotted in its own subplot.
    xlabels : str or list of str, optional
        Label(s) for the x-axis. Can be a single string (applied to all) or a list of length N.
    ylabels : str or list of str, optional
        Label(s) for the y-axis.
    zlabels : str or list of str, optional
        Label(s) for the z-axis (only for 3D plots).
    **kwargs : dict
        Additional arguments passed to `ax.plot()`.

    Returns
    -------
    list of matplotlib.axes.Axes
        List of Axes objects.

    Notes
    -----
    - If `separate_axes` is False, all trajectories are plotted together and a single legend is shown.
    - If multiple subplots are used in 3D mode, views are synchronized.
    """
    if data.ndim == 2:
        data = data[np.newaxis, ...]  # Ensure (N, T, 3) shape for single sample

    N, T, D = data.shape
    if D != 3:
        raise ValueError("Data must have shape (N, T, 3) for 3D parametric plotting.")

    # Prepare labels
    sample_labels = sample_labels or [f"Sample {i+1}" for i in range(N)]
    if len(sample_labels) != N:
        raise ValueError("Length of sample_names must match the number of samples N.")

    if isinstance(xlabels, str):
        xlabels = [xlabels] * N
    elif len(xlabels) != N:
        raise ValueError(
            f"Length of xlabels must match the number of samples {N}, got {len(xlabels)}."
        )

    if isinstance(ylabels, str):
        ylabels = [ylabels] * N
    elif len(ylabels) != N:
        raise ValueError(
            f"Length of ylabels must match the number of samples {N}, got {len(ylabels)}."
        )

    if isinstance(zlabels, str):
        zlabels = [zlabels] * N
    elif len(zlabels) != N:
        raise ValueError(
            f"Length of zlabels must match the number of samples {N}, got {len(zlabels)}."
        )

    # Plotting
    if separate_axes:
        cols = min(N, 3)
        rows = int(np.ceil(N / cols))
        fig, axes = plt.subplots(
            rows, cols, figsize=(cols * 5, rows * 5), subplot_kw={"projection": "3d"}
        )
        axes = np.ravel(axes)

        for i in range(N, len(axes)):  # Hide unused axes
            fig.delaxes(axes[i])

        if suptitle:
            fig.suptitle(suptitle, fontsize=14)
            fig.subplots_adjust(top=0.85)  # Pushes plots downward
    else:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="3d")
        axes = [ax] * N
        if suptitle:
            fig.suptitle(suptitle, fontsize=14, y=0.95)
            fig.subplots_adjust(top=0.80)

    for i in range(N):
        ax = axes[i]  # Select correct subplot
        ax.plot(
            data[i, :, 0],
            data[i, :, 1],
            data[i, :, 2],
            label=sample_labels[i],
            **kwargs,
        )

        if separate_axes and N > 1:
            ax.set_title(sample_labels[i])

        ax.set_xlabel(xlabels[i])
        ax.set_ylabel(ylabels[i])
        ax.set_zlabel(zlabels[i])

    if not separate_axes:  # Show legend only when applicable
        axes[0].legend()

    if separate_axes:
        plt.tight_layout()

        # Synchronize 3D rotation
        def on_move(event):
            if event.inaxes not in axes:
                return
            elev, azim = event.inaxes.elev, event.inaxes.azim
            for ax in axes:
                if (ax.elev, ax.azim) != (elev, azim):  # Update only if changed
                    ax.view_init(elev=elev, azim=azim)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", on_move)

    return axes


def plot_heatmap(
    data: np.ndarray,
    dt: float = 1.0,
    lam: float = 1.0,
    suptitle: Optional[str] = None,
    sample_labels: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    cmap: str = "viridis",
    aspect: Union[str, float] = "auto",
    colorbar_label: str = "Feature Value",
    xlabels: Optional[Union[str, List[str]]] = None,
    ylabels: Optional[Union[str, List[str]]] = None,
    xlims: Optional[Tuple[float, float]] = None,
    ylims: Optional[Tuple[float, float]] = None,
    transpose: bool = True,
    **kwargs,
):
    """
    Plots a heatmap representation of 2D time-series data, optionally
    stacking multiple samples as subplots.

    The x-axis represents time, scaled using `dt` and `lam`, while the y-axis
    corresponds to features. Each row in the heatmap represents a feature, and
    each column represents a time step.

    Parameters
    ----------
    data : np.ndarray
        Input array of shape (T, D) or (N, T, D).
        - T: Number of time steps.
        - D: Number of features.
        - N: Number of samples (optional).
    dt : float, optional
        Time step for scaling the x-axis (default=1.0).
    lam : float, optional
        Additional scaling factor for the x-axis (default=1.0).
    suptitle : str, optional
        Title for the entire figure.
    sample_labels : list of str, optional
        Titles for individual subplots (length N).
    figsize : tuple of float, optional
        Figure size (width, height).
    cmap : str, optional
        Colormap for the heatmap (default='viridis').
    aspect : str or float, optional
        Aspect ratio for `imshow()` (default='auto').
    colorbar_label : str, optional
        Label for the colorbar (default="Feature Value").
    xlabels : str or list of str, optional
        Labels for the x-axis. If multiple samples exist, this can be a list.
    ylabels : str or list of str, optional
        Labels for the y-axis. Can be a single string or a list (one per sample).
    xlims : tuple of float, optional
        Limits for the x-axis.
    ylims : tuple of float, optional
        Limits for the y-axis.
    transpose : bool, optional
        Whether to transpose the heatmap data (default=True).
    **kwargs : dict
        Additional keyword arguments passed to `imshow()` (e.g., `vmin`, `vmax`).

    Returns
    -------
    list of matplotlib.axes.Axes
        Axes objects corresponding to the subplots.

    Notes
    -----
    - If `data` has shape (T, D), a single heatmap is plotted.
    - If `data` has shape (N, T, D), multiple heatmaps are stacked as subplots.
    - By default, data is transposed so that time runs horizontally and features run vertically.
    - The x-axis is automatically scaled using `dt * lam`.
    - If `sample_labels` is provided, each subplot gets an individual title.
    - A colorbar is displayed to the right of the heatmaps, common to all subplots for comparison.
    """
    # Ensure we have shape (N, T, D)
    if data.ndim == 1:
        raise ValueError("Data must have shape (T, D) or (N, T, D).")

    if data.ndim == 2:
        data = data[np.newaxis, ...]  # Becomes (1, T, D)

    if data.ndim > 3:
        raise ValueError("data must have shape (T, D) or (N, T, D).")

    N, T, D = data.shape

    # Generate default labels if not provided
    sample_labels = (
        sample_labels if sample_labels else [f"Sample {i+1}" for i in range(N)]
    )
    if len(sample_labels) != N:
        raise ValueError(
            f"Length of sample_labels must match the number of samples N, got {len(sample_labels)}."
        )

    if isinstance(xlabels, str) or xlabels is None:
        # No labels for intermediate samples. Single label for last sample.
        xlabels = [None] * (N - 1) + [xlabels or "Time"]

    elif xlabels and len(xlabels) != N:
        raise ValueError(
            f"Length of xlabels must match the number of samples N, got {len(xlabels)}."
        )

    if isinstance(ylabels, str) or ylabels is None:
        ylabels = [ylabels or "Y"] * N

    elif ylabels and len(ylabels) != N:
        raise ValueError(
            f"Length of ylabels must match the number of samples N, got {len(ylabels)}."
        )

    # Auto-size figure if needed
    figsize = figsize or (12, min(3 * N, 15))  # No more than 5 plot-heights

    # Create subplots with N rows, 1 column, and adjust layout for colorbar
    fig, axes = plt.subplots(
        nrows=N, ncols=1, figsize=figsize, sharex=True, sharey=True
    )
    fig.subplots_adjust(hspace=0.4)  # Increase vertical spacing

    axes = np.atleast_1d(axes)  # Ensure iterable structure for consistency

    # Set x-axis scaling
    x_min, x_max = xlims or (0.0, float(T) * dt * lam)
    y_min, y_max = ylims or (0.0, float(D))

    extent = [x_min, x_max, y_min, y_max]

    im = None
    for i in range(N):
        ax = axes[i]

        # Transpose data for imshow: (T, D) → (D, T)
        if transpose:
            data_i = data[i].T
        else:
            data_i = data[i]

        im = ax.imshow(
            data_i,
            cmap=cmap,
            aspect=aspect,
            extent=extent,
            **kwargs,
        )

        # Set subplot labels
        ax.set_ylabel(ylabels[i])

        if xlabels[i] is not None:
            ax.set_xlabel(xlabels[i])

        ax.set_title(sample_labels[i])

    # Adjust layout to reserve space for the colorbar
    fig.subplots_adjust(right=0.85)  # Leave space on the right

    # Add colorbar in the reserved space
    cbar_ax = fig.add_axes([0.88, 0.2, 0.01, 0.6])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax)

    # Set colorbar label
    cbar.set_label(colorbar_label)

    # Adjust suptitle position
    if suptitle:
        # fig.suptitle(t=suptitle, fontsize=14, y=1.02)
        fig.suptitle(suptitle, fontsize=14, y=1.05)
        plt.subplots_adjust(top=0.88)  # Adjust top margin to make space for suptitle

    return axes


__all__ = [
    "plot_2d_timeseries",
    "plot_3d_parametric",
    "plot_heatmap",
]

def __dir__():
    return __all__