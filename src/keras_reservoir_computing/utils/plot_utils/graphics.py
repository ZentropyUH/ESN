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
) -> List[plt.Axes]:
    """
    Plot 2D time series data with optional multiple samples and multiple features.

    This function can handle data of shape:
    - (T, D), interpreted as one sample with T time steps and D features, or
    - (N, T, D), interpreted as N samples, each with T time steps and D features.

    Parameters
    ----------
    data : np.ndarray
        The input array of shape (N, T, D) or (T, D).
        - N: Number of samples (optional).
        - T: Number of time steps.
        - D: Number of features.
    dt : float, optional
        Time step between consecutive data points (default 1.0).
    lam : float, optional
        Additional scaling factor for the time axis (default 1.0).
    suptitle : str, optional
        Title for the entire figure.
    sample_labels : list of str, optional
        Labels for each of the N samples. Must be length N if provided.
        If None, samples are labeled "Sample 1", "Sample 2", etc.
    feature_labels : list of str, optional
        Labels for each of the D features. Must be length D if provided.
        If None, features are labeled "Feature 1", "Feature 2", etc.
    xlabels : str or list of str, optional
        Label(s) for the x-axis. If a single string, it is applied to each feature plot.
        If a list, must be length D. Defaults to "Time".
    ylabels : str or list of str, optional
        Label(s) for the y-axis. If a single string, it is applied to each feature plot.
        If a list, must be length D. Defaults to "Feature Value".
    separate_axes : bool, optional
        If True, each feature is plotted on a separate axis (vertical stack). Otherwise,
        all features are plotted in a single subplot (default=False).
    yscale : {'linear', 'log'}, optional
        Scale for the y-axis (default='linear').
    xscale : {'linear', 'log'}, optional
        Scale for the x-axis (default='linear').
    **kwargs : dict
        Additional keyword arguments passed to :func:`plt.plot`.

    Returns
    -------
    list of matplotlib.axes.Axes
        A list of Axes objects corresponding to the created plots (one per feature).

    Raises
    ------
    ValueError
        If data dimensionality is greater than 3, or if label lengths don't match
        the data's N or D dimensions.

    Notes
    -----
    - If multiple samples share a single plot (`separate_axes=False`), the legend
      distinguishes them by color and a label that combines the sample and feature names.
    - If `separate_axes=True`, each feature is plotted in a separate subplot (vertical).
      If multiple samples exist, each subplot has lines for all N samples.
    - The time axis is computed as `t = np.arange(T) * dt * lam`.
    """
    # Validate or reshape data into (N, T, D)
    if data.ndim > 3:
        raise ValueError(f"Data must have ndim <= 3, got {data.ndim}.")
    if data.ndim == 1:
        # Force shape (1, T, 1)
        data = data[np.newaxis, ..., np.newaxis]
    if data.ndim == 2:
        # Force shape (1, T, D)
        data = data[np.newaxis, ...]

    N, T, D = data.shape

    # Time axis
    t = np.arange(T) * dt * lam

    # Prepare figure/axes
    if separate_axes:
        fig, axes = plt.subplots(
            D, 1, figsize=(12, 2 * D), sharex=True
        )  # one subplot per feature
    else:
        fig = plt.figure(figsize=(12, 3))
        axes = plt.gca()  # single subplot for all features

    axes = np.atleast_1d(axes)  # Make sure axes is array-like

    if suptitle:
        fig.suptitle(suptitle, fontsize=14)

    # Default sample/feature labels
    sample_labels = sample_labels or [f"Sample {i+1}" for i in range(N)]
    feature_labels = feature_labels or [f"Feature {j+1}" for j in range(D)]

    if len(sample_labels) != N:
        raise ValueError(
            f"Length of sample_labels must be {N}, got {len(sample_labels)}."
        )
    if len(feature_labels) != D:
        raise ValueError(
            f"Length of feature_labels must be {D}, got {len(feature_labels)}."
        )

    # Handle xlabels, ylabels
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
            # Combine labels depending on whether features share an axis
            if N > 1 and not separate_axes:
                label = f"{sample_labels[sample_idx]} - {feature_labels[feature_idx]}"
            else:
                # Single-sample or separate-axes mode
                label = (
                    sample_labels[sample_idx]
                    if N > 1 and separate_axes
                    else feature_labels[feature_idx]
                )

            ax.plot(t, data[sample_idx, :, feature_idx], label=label, **kwargs)

        # Axis scales and labels
        ax.set_yscale(yscale)
        ax.set_xscale(xscale)
        ax.set_ylabel(ylabels[feature_idx])
        ax.set_xlabel(xlabels[feature_idx])
        ax.set_title(feature_labels[feature_idx])

    # Handle legends and layout
    if separate_axes:
        for ax in axes:
            ax.legend(loc="upper left")
        plt.tight_layout()
    else:
        handles, legend_labels = axes[0].get_legend_handles_labels()
        if N == 1:
            # Only one sample
            axes[0].legend(handles, legend_labels, loc="upper left", frameon=True)
        else:
            # Multiple samples in a single plot
            # Group legends by sample_labels
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

    return list(axes)


def plot_3d_parametric(
    data: np.ndarray,
    suptitle: Optional[str] = None,
    sample_labels: Optional[List[str]] = None,
    separate_axes: bool = False,
    xlabels: Union[str, List[str]] = "X",
    ylabels: Union[str, List[str]] = "Y",
    zlabels: Union[str, List[str]] = "Z",
    **kwargs,
) -> List[plt.Axes]:
    """
    Plot 3D parametric curves (trajectories) from input data.

    Each trajectory represents (x(t), y(t), z(t)) over time. The function
    supports multiple trajectories and can plot them either together on a
    single axis or on separate subplots.

    Parameters
    ----------
    data : np.ndarray
        Input array of shape (N, T, 3) or (T, 3).
        - N: Number of trajectories (optional).
        - T: Number of time steps.
        - 3: The three spatial dimensions (x, y, z).
    suptitle : str, optional
        Overall title for the figure.
    sample_labels : list of str, optional
        Labels for each trajectory (length N). If None, defaults to "Sample 1", etc.
    separate_axes : bool, optional
        If True, each trajectory is plotted on its own 3D subplot.
        Otherwise, all trajectories share the same 3D axis (default=False).
    xlabels : str or list of str, optional
        x-axis label(s). If a single string, it’s reused for all subplots.
        If a list, must be length N.
    ylabels : str or list of str, optional
        y-axis label(s). Same rules as xlabels.
    zlabels : str or list of str, optional
        z-axis label(s). Same rules as xlabels.
    **kwargs : dict
        Additional keyword arguments passed to :func:`Axes.plot`.

    Returns
    -------
    list of matplotlib.axes.Axes
        A list of 3D Axes objects where the trajectories are drawn.

    Raises
    ------
    ValueError
        If data is not (T, 3) or (N, T, 3), or if label lengths don’t match N.

    Notes
    -----
    - If multiple 3D subplots are used, their rotation (view angle) is synchronized.
    - Legend is only shown if `separate_axes=False`.
    """
    # Ensure data is shape (N, T, 3)
    if data.ndim == 2:
        # Single trajectory
        data = data[np.newaxis, ...]
    N, T, D = data.shape
    if D != 3:
        raise ValueError("Data must have shape (N, T, 3) or (T, 3) for 3D plotting.")

    # Prepare default sample labels
    sample_labels = sample_labels or [f"Sample {i+1}" for i in range(N)]
    if len(sample_labels) != N:
        raise ValueError(
            f"Length of sample_labels ({len(sample_labels)}) must match N={N}."
        )

    # Expand xlabels, ylabels, zlabels if needed
    if isinstance(xlabels, str):
        xlabels = [xlabels] * N
    elif len(xlabels) != N:
        raise ValueError(f"Expected {N} xlabels, got {len(xlabels)}.")

    if isinstance(ylabels, str):
        ylabels = [ylabels] * N
    elif len(ylabels) != N:
        raise ValueError(f"Expected {N} ylabels, got {len(ylabels)}.")

    if isinstance(zlabels, str):
        zlabels = [zlabels] * N
    elif len(zlabels) != N:
        raise ValueError(f"Expected {N} zlabels, got {len(zlabels)}.")

    # Create figure & axes
    if separate_axes:
        cols = min(N, 3)
        rows = int(np.ceil(N / cols))
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(cols * 5, rows * 5),
            subplot_kw={"projection": "3d"},
        )
        axes = np.ravel(axes)
        for i in range(N, len(axes)):
            # Hide unused
            fig.delaxes(axes[i])
        if suptitle:
            fig.suptitle(suptitle, fontsize=14)
            fig.subplots_adjust(top=0.85)
    else:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="3d")
        axes = [ax] * N
        if suptitle:
            fig.suptitle(suptitle, fontsize=14, y=0.95)
            fig.subplots_adjust(top=0.80)

    for i in range(N):
        ax_i = axes[i]
        ax_i.plot(
            data[i, :, 0],
            data[i, :, 1],
            data[i, :, 2],
            label=sample_labels[i],
            **kwargs,
        )
        if separate_axes and N > 1:
            ax_i.set_title(sample_labels[i])

        ax_i.set_xlabel(xlabels[i])
        ax_i.set_ylabel(ylabels[i])
        ax_i.set_zlabel(zlabels[i])

    if not separate_axes:
        # Show a single legend if everything is on one axis
        axes[0].legend(loc="best")

    if separate_axes:
        plt.tight_layout()

        # Synchronize 3D rotation
        def on_move(event):
            if event.inaxes not in axes:
                return
            elev, azim = event.inaxes.elev, event.inaxes.azim
            for ax_i in axes:
                if (ax_i.elev, ax_i.azim) != (elev, azim):
                    ax_i.view_init(elev=elev, azim=azim)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", on_move)

    return list(axes)


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
) -> List[plt.Axes]:
    """
    Plot a heatmap representation of time-series data (or multiple samples of data).

    The data can be:
     - (T, D) shape for a single sample with T time steps and D features.
     - (N, T, D) shape for N samples, each with T time steps and D features.

    By default, the data is transposed for plotting so that time runs along the
    x-axis and feature index runs along the y-axis.

    Parameters
    ----------
    data : np.ndarray
        The input data array of shape (T, D) or (N, T, D).
    dt : float, optional
        Time step for scaling the x-axis (default=1.0).
    lam : float, optional
        Additional scaling factor for the x-axis (default=1.0).
    suptitle : str, optional
        A title for the entire figure.
    sample_labels : list of str, optional
        Titles for the subplots if N > 1. Must be length N if provided.
    figsize : tuple of float, optional
        Size of the figure (width, height). Defaults to something reasonable if None.
    cmap : str, optional
        Colormap used by :func:`plt.imshow`. Default 'viridis'.
    aspect : str or float, optional
        The aspect ratio passed to `imshow`. Default 'auto'.
    colorbar_label : str, optional
        Label for the colorbar. Default "Feature Value".
    xlabels : str or list of str, optional
        Label(s) for the x-axis. If a single string or None, it's used for the last subplot.
        If a list, must have length = N.
    ylabels : str or list of str, optional
        Label(s) for the y-axis. If a single string or None, it's used for each subplot.
        If a list, must have length = N.
    xlims : tuple of float, optional
        (min_x, max_x) axis limits for the x-axis.
    ylims : tuple of float, optional
        (min_y, max_y) axis limits for the y-axis.
    transpose : bool, optional
        If True, data is transposed so that time is horizontal and features are vertical.
        Defaults to True.
    **kwargs : dict
        Additional keyword args passed to `imshow`, e.g., `vmin`, `vmax`.

    Returns
    -------
    list of matplotlib.axes.Axes
        A list of Axes objects, one per sample.

    Raises
    ------
    ValueError
        If data shape is invalid or if label lists don't match N.

    Notes
    -----
    - Each sample is plotted in a separate subplot (vertical stack).
    - The horizontal axis runs from 0..T*dt*lam (unless xlims is given).
    - A single colorbar is placed to the right of all subplots.
    - If `transpose=True`, the shape is displayed as (D, T) in the heatmap.
      Otherwise, (T, D).
    """
    # Ensure shape (N, T, D)
    if data.ndim == 1:
        raise ValueError("Data must have shape (T, D) or (N, T, D), not 1D.")

    if data.ndim == 2:
        data = data[np.newaxis, ...]  # (1, T, D)

    if data.ndim > 3:
        raise ValueError("Data must have shape (T, D) or (N, T, D).")

    N, T, D = data.shape

    # Prepare default labels
    sample_labels = sample_labels or [f"Sample {i+1}" for i in range(N)]
    if len(sample_labels) != N:
        raise ValueError(
            f"Length of sample_labels must match N={N}, got {len(sample_labels)}."
        )

    # Manage xlabels
    if isinstance(xlabels, str) or xlabels is None:
        # This label is only placed on the final subplot
        xlabels = [None] * (N - 1) + [xlabels or "Time"]
    elif len(xlabels) != N:
        raise ValueError(f"Length of xlabels must match N={N}, got {len(xlabels)}.")

    # Manage ylabels
    if isinstance(ylabels, str) or ylabels is None:
        ylabels = [ylabels or "Y"] * N
    elif len(ylabels) != N:
        raise ValueError(f"Length of ylabels must match N={N}, got {len(ylabels)}.")

    # Figure size
    if figsize is None:
        # Limit total figure height to ~15
        figsize = (12, min(3 * N, 15))

    fig, axes = plt.subplots(
        nrows=N, ncols=1, figsize=figsize, sharex=True, sharey=True
    )
    fig.subplots_adjust(hspace=0.4)

    # Scale time axis
    x_min, x_max = xlims or (0.0, float(T) * dt * lam)
    y_min, y_max = ylims or (0.0, float(D))

    # extent for imshow
    extent = [x_min, x_max, y_min, y_max]

    axes = np.atleast_1d(axes)
    im = None

    for i in range(N):
        ax = axes[i]

        # Possibly transpose (T, D) -> (D, T)
        if transpose:
            data_i = data[i].T  # shape (D, T)
        else:
            data_i = data[i]  # shape (T, D)

        im = ax.imshow(
            data_i,
            cmap=cmap,
            aspect=aspect,
            extent=extent,
            **kwargs,
        )

        ax.set_title(sample_labels[i])
        ax.set_ylabel(ylabels[i])
        # Only the last subplot gets the x-axis label
        if xlabels[i] is not None:
            ax.set_xlabel(xlabels[i])

    # Space on right for colorbar
    fig.subplots_adjust(right=0.85)
    # Add colorbar
    cbar_ax = fig.add_axes([0.88, 0.2, 0.01, 0.6])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(colorbar_label)

    # Adjust suptitle
    if suptitle:
        fig.suptitle(suptitle, fontsize=14, y=1.05)
        plt.subplots_adjust(top=0.88)

    return list(axes)


__all__ = [
    "plot_2d_timeseries",
    "plot_3d_parametric",
    "plot_heatmap",
]


def __dir__():
    return __all__
