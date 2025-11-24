from typing import Callable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from .helpers import (
    _timeseries_base_plot,
    _delay_transform,
    _relative_extremes,
    _poincare_section,
    _recurrence_plot,
    plot_2d,
    scatter_2d,
)
from functools import partial


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
    figsize: Optional[Tuple[int, int]] = None,
    sharex: bool = True,
    yscale: str = "linear",
    xscale: str = "linear",
    data_transform: Optional[Callable] = None,
    **kwargs,
) -> List[plt.Axes]:
    """
    Makes a 2D time-series plot of the input data.

    This function is a specialized wrapper around `_timeseries_base_plot`,
    automatically setting `plot_func=plot_2d`.

    See Also
    --------
    :func:`_timeseries_base_plot` : The base function for time series plotting.
    """
    return _timeseries_base_plot(
        data=data,
        dt=dt,
        lam=lam,
        suptitle=suptitle,
        sample_labels=sample_labels,
        feature_labels=feature_labels,
        xlabels=xlabels,
        ylabels=ylabels,
        separate_axes=separate_axes,
        figsize=figsize,
        sharex=sharex,
        yscale=yscale,
        xscale=xscale,
        data_transform=data_transform,
        plot_func=plot_2d,
        **kwargs,
    )


def scatter_2d_timeseries(
    data: np.ndarray,
    dt: float = 1.0,
    lam: float = 1.0,
    suptitle: Optional[str] = None,
    sample_labels: Optional[List[str]] = None,
    feature_labels: Optional[List[str]] = None,
    xlabels: Union[str, List[str]] = "Time",
    ylabels: Union[str, List[str]] = "Feature Value",
    separate_axes: bool = False,
    figsize: Optional[Tuple[int, int]] = None,
    sharex: bool = True,
    yscale: str = "linear",
    xscale: str = "linear",
    data_transform: Optional[Callable] = None,
    **kwargs,
) -> List[plt.Axes]:
    """
    Makes a 2D scatter plot of the input data.

    This function is a specialized wrapper around `_timeseries_base_plot`,
    automatically setting `plot_func=scatter_2d`.

    See Also
    --------
    :func:`_timeseries_base_plot` : The base function for time series plotting.
    """
    return _timeseries_base_plot(
        data=data,
        dt=dt,
        lam=lam,
        suptitle=suptitle,
        sample_labels=sample_labels,
        feature_labels=feature_labels,
        xlabels=xlabels,
        ylabels=ylabels,
        separate_axes=separate_axes,
        figsize=figsize,
        sharex=sharex,
        yscale=yscale,
        xscale=xscale,
        data_transform=data_transform,
        plot_func=scatter_2d,
        **kwargs,
    )


def plot_delay_map(
    data: np.ndarray,
    delay_time: float = 1.0,
    dt: float = 1.0,
    lam: float = 1.0,
    suptitle: Optional[str] = None,
    sample_labels: Optional[List[str]] = None,
    feature_labels: Optional[List[str]] = None,
    xlabels: Union[str, List[str]] = "x(t)",
    ylabels: Union[str, List[str]] = "x(t+τ)",
    separate_axes: bool = False,
    figsize: Optional[Tuple[int, int]] = None,
    sharex: bool = True,
    yscale: str = "linear",
    xscale: str = "linear",
    **kwargs,
) -> List[plt.Axes]:
    """
    Makes a delay map plot of the input data.

    This function applies a delay transformation to the data and then plots
    the delayed values against the original values.

    See Also
    --------
    :func:`_delay_transform` : The delay transformation function.
    """
    return _timeseries_base_plot(
        data=data,
        dt=dt,
        lam=lam,
        suptitle=suptitle,
        sample_labels=sample_labels,
        feature_labels=feature_labels,
        xlabels=xlabels,
        ylabels=ylabels,
        separate_axes=separate_axes,
        figsize=figsize,
        sharex=sharex,
        yscale=yscale,
        xscale=xscale,
        data_transform=partial(_delay_transform, delay_time=delay_time),
        plot_func=scatter_2d,
        **kwargs,
    )


def plot_extremes_map(
    data: np.ndarray,
    dt: float = 1.0,
    lam: float = 1.0,
    extreme_type: str = "max",
    suptitle: Optional[str] = None,
    sample_labels: Optional[List[str]] = None,
    feature_labels: Optional[List[str]] = None,
    xlabels: Union[str, List[str]] = "x(n)",
    ylabels: Union[str, List[str]] = "x(n+1)",
    separate_axes: bool = False,
    figsize: Optional[Tuple[int, int]] = None,
    sharex: bool = True,
    yscale: str = "linear",
    xscale: str = "linear",
    **kwargs,
) -> List[plt.Axes]:
    """
    Makes a return map plot of the relative extremes of the input data.

    This function applies a relative extremes transformation to the data and then
    plots the next extreme value against the current one.

    See Also
    --------
    :func:`_relative_extremes` : The relative extremes transformation function.
    """
    return _timeseries_base_plot(
        data=data,
        dt=dt,
        lam=lam,
        suptitle=suptitle,
        sample_labels=sample_labels,
        feature_labels=feature_labels,
        xlabels=xlabels,
        ylabels=ylabels,
        separate_axes=separate_axes,
        figsize=figsize,
        sharex=sharex,
        yscale=yscale,
        xscale=xscale,
        data_transform=partial(_relative_extremes, extreme_type=extreme_type),
        plot_func=scatter_2d,
        **kwargs,
    )


def plot_poincare_section(
    data: np.ndarray,
    dt: float = 1.0,
    lam: float = 1.0,
    plane: str = "x=0",
    direction: str = "positive",
    suptitle: Optional[str] = None,
    sample_labels: Optional[List[str]] = None,
    feature_labels: Optional[List[str]] = None,
    xlabels: Union[str, List[str]] = "y",
    ylabels: Union[str, List[str]] = "z",
    separate_axes: bool = False,
    figsize: Optional[Tuple[int, int]] = None,
    sharex: bool = True,
    yscale: str = "linear",
    xscale: str = "linear",
    **kwargs,
) -> List[plt.Axes]:
    """
    Makes a Poincaré section plot of the input data.

    This function applies a Poincaré section transformation to the data and then
    plots the resulting points.

    See Also
    --------
    :func:`_poincare_section` : The Poincaré section transformation function.
    """
    return _timeseries_base_plot(
        data=data,
        dt=dt,
        lam=lam,
        suptitle=suptitle,
        sample_labels=sample_labels,
        feature_labels=feature_labels,
        xlabels=xlabels,
        ylabels=ylabels,
        separate_axes=separate_axes,
        figsize=figsize,
        sharex=sharex,
        yscale=yscale,
        xscale=xscale,
        data_transform=partial(_poincare_section, plane=plane, direction=direction),
        plot_func=scatter_2d,
        **kwargs,
    )


def plot_recurrence(
    data: np.ndarray,
    threshold: float = 0.1,
    metric: str = "euclidean",
    suptitle: Optional[str] = None,
    sample_labels: Optional[List[str]] = None,
    figsize: Optional[Tuple[int, int]] = None,
    cmap: str = "binary",
    **kwargs,
) -> List[plt.Axes]:
    """
    Makes a recurrence plot of the input data.

    This function applies a recurrence plot transformation to the data and then
    displays the resulting matrix as an image.

    Parameters
    ----------
    data : np.ndarray
        The input data array of shape (N, T, D).
    threshold : float, optional
        The threshold for considering points as recurrent.
    metric : str, optional
        The distance metric to use. Options: "euclidean", "manhattan", "chebyshev".
    suptitle : str, optional
        The super title for the entire figure.
    sample_labels : list of str, optional
        Labels for each sample.
    figsize : tuple of int, optional
        The size of the figure.
    cmap : str, optional
        The colormap to use for the recurrence plot.
    **kwargs
        Additional keyword arguments passed to `imshow`.

    Returns
    -------
    list of plt.Axes
        A list of matplotlib axes objects.

    See Also
    --------
    :func:`_recurrence_plot` : The recurrence plot transformation function.
    """
    # Compute recurrence plot
    recurrence = _recurrence_plot(data, threshold=threshold, metric=metric)

    # Prepare figure
    N = recurrence.shape[0]
    if figsize is None:
        figsize = (6, 6 * N)

    fig, axes = plt.subplots(N, 1, figsize=figsize)
    axes = np.atleast_1d(axes)

    if suptitle:
        fig.suptitle(suptitle, fontsize=14)

    # Prepare default labels
    sample_labels = sample_labels or [f"Sample {i+1}" for i in range(N)]
    if len(sample_labels) != N:
        raise ValueError(f"Length of sample_labels must be {N}, got {len(sample_labels)}.")

    # Plot each recurrence plot
    for i, ax in enumerate(axes):
        im = ax.imshow(recurrence[i], cmap=cmap, **kwargs)
        ax.set_title(sample_labels[i])
        ax.set_xlabel("Time")
        ax.set_ylabel("Time")

    # Add colorbar
    fig.colorbar(im, ax=axes.ravel().tolist())

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
        x-axis label(s). If a single string, it's reused for all subplots.
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
        If data is not (T, 3) or (N, T, 3), or if label lengths don't match N.
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
        raise ValueError(f"Length of sample_labels ({len(sample_labels)}) must match N={N}.")

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
        raise ValueError(f"Length of sample_labels must match N={N}, got {len(sample_labels)}.")

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

    fig, axes = plt.subplots(nrows=N, ncols=1, figsize=figsize, sharex=True, sharey=True)
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
    "scatter_2d_timeseries",
    "plot_delay_map",
    "plot_extremes_map",
    "plot_poincare_section",
    "plot_recurrence",
    "plot_3d_parametric",
    "plot_heatmap",
]


def __dir__():
    return __all__
