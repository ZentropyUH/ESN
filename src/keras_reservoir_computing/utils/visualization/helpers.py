import numpy as np
from scipy.signal import argrelmax, argrelmin
from typing import Tuple, List, Optional, Union, Callable
import matplotlib.pyplot as plt


# Base plotting function
def _timeseries_base_plot(
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
    sharex=True,
    yscale: str = "linear",
    xscale: str = "linear",
    data_transform: Optional[Callable] = None,
    plot_func: Callable = plt.plot,
    **kwargs,
) -> List[plt.Axes]:
    """
    Generic scaffold for 2D time-series plotting.
    It takes a 'plot_func' callback to handle how the data is actually plotted.

    Parameters
    ----------
    data : np.ndarray
        The time-series data to plot. It should be of shape (N, T, D) where
        N is the number of samples, T is the number of time points, and D is the number of features.
    dt : float, optional
        The time step between consecutive data points. Default is 1.0.
    lam : float, optional
        A scaling factor for the time axis. Default is 1.0.
    suptitle : str, optional
        The super title for the entire figure. Default is None.
    sample_labels : list of str, optional
        Labels for each sample. Default is None, which will generate labels as "Sample 1", "Sample 2", etc.
    feature_labels : list of str, optional
        Labels for each feature. Default is None, which will generate labels as "Feature 1", "Feature 2", etc.
    xlabels : str or list of str, optional
        Labels for the x-axis. If a single string is provided, it will be used for all features. Default is "Time".
    ylabels : str or list of str, optional
        Labels for the y-axis. If a single string is provided, it will be used for all features. Default is "Feature Value".
    separate_axes : bool, optional
        If True, each feature will be plotted on a separate axis. Default is False.
    figsize : tuple of int, optional
        The size of the figure. Default is None, which will use (12, 3) for combined axes or (12, 3 * D) for separate axes.
    sharex : bool, optional
        Whether to share the x-axis across subplots. Default is True.
    yscale : str, optional
        The scale for the y-axis. Default is "linear". Possible values are "linear", "log", "symlog", "logit".
    xscale : str, optional
        The scale for the x-axis. Default is "linear". Possible values are "linear", "log", "symlog", "logit".
    data_transform : Callable, optional
        A function that, if provided, is called on the data before plotting.
        This can implement time delays, maxima detection, or anything else.
        It must have the signature:
        `data_transform(data: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]`.
    plot_func : Callable, optional
        The plot function to use for plotting the data. It must have the signature:
        `plot_func(ax: plt.Axes, t: np.ndarray, y: np.ndarray, label: str, **kwargs) -> None`.
    **kwargs
        Additional keyword arguments passed on to `plot_func`.

    Returns
    -------
    list of plt.Axes
        A list of matplotlib axes objects.
    """
    # -- Step 1. Validate or reshape data into (N, T, D) --
    if data.ndim > 3:
        raise ValueError(f"Data must have ndim <= 3, got {data.ndim}.")
    if data.ndim == 1:
        data = data[np.newaxis, ..., np.newaxis]
    if data.ndim == 2:
        data = data[np.newaxis, ...]

    N, T, D = data.shape

    t = np.arange(T) * dt * lam

    # -- Step 3. Prepare figure/axes --
    if separate_axes:
        fig, axes = plt.subplots(D, 1, figsize=figsize or (12, 3 * D), sharex=sharex)
        axes = np.atleast_1d(axes)
    else:
        fig = plt.figure(figsize=figsize or (12, 3))
        ax_single = plt.gca()
        axes = [ax_single] * D

    if suptitle:
        fig.suptitle(suptitle, fontsize=14)

    # -- Step 4. Handle default sample/feature labels, xlabels, ylabels --
    sample_labels = sample_labels or [f"Sample {i+1}" for i in range(N)]
    feature_labels = feature_labels or [f"Feature {j+1}" for j in range(D)]
    if len(sample_labels) != N:
        raise ValueError(f"Length of sample_labels must be {N}, got {len(sample_labels)}.")
    if len(feature_labels) != D:
        raise ValueError(f"Length of feature_labels must be {D}, got {len(feature_labels)}.")

    if isinstance(xlabels, str):
        xlabels = [xlabels] * D
    elif len(xlabels) != D:
        raise ValueError(f"Expected {D} x-labels, got {len(xlabels)}.")

    if isinstance(ylabels, str):
        ylabels = [ylabels] * D
    elif len(ylabels) != D:
        raise ValueError(f"Expected {D} y-labels, got {len(ylabels)}.")

    # -- Step 5. Main loop to plot each feature across N samples --
    for feature_idx, ax in enumerate(axes):
        N_new, T_new, D_new = data.shape

        for sample_idx in range(N_new):
            if N_new > 1 and not separate_axes:
                label = f"{sample_labels[sample_idx]} - {feature_labels[feature_idx]}"
            else:
                label = (
                    sample_labels[sample_idx]
                    if (N_new > 1 and separate_axes)
                    else feature_labels[feature_idx]
                )

            if data_transform is not None:
                t, y_vals = data_transform(data[sample_idx, :, feature_idx], dt)
            else:
                y_vals = data[sample_idx, :, feature_idx]

            plot_func(ax, t, y_vals, label=label, **kwargs)

        if separate_axes and N > 1:
            ax.set_title(feature_labels[feature_idx])

        ax.set_yscale(yscale)
        ax.set_xscale(xscale)
        ax.set_ylabel(ylabels[feature_idx])
        ax.set_xlabel(xlabels[feature_idx])

    # -- Step 6. Legends, layout, etc. --
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

    return list(axes)


# Transformation functions
def _delay_transform(
    data: np.ndarray, dt: float, delay_time: float = 5.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a delay transformation to the data.

    Parameters
    ----------
    data : np.ndarray
        The input data array of shape (T,) or (N, T, D).
    dt : float
        The time step between data points.
    delay_time : float, optional
        The delay time to apply, by default 5.0.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing the delayed data and the data without delay.
    """
    delay_steps = int(delay_time / dt)

    if data.ndim == 1:
        delayed_data = data[delay_steps:]
        data_without_delay = data[:-delay_steps]
    else:
        # Handle 3D arrays (N, T, D)
        delayed_data = data[:, delay_steps:, :]
        data_without_delay = data[:, :-delay_steps, :]

    return delayed_data, data_without_delay


def _relative_extremes(
    data: np.ndarray, dt: float, extreme_type: str = "max"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a relative extremes transformation to the data.

    Parameters
    ----------
    data : np.ndarray
        The input data array of shape (T,) or (N, T, D).
    dt : float
        The time step between data points.
    extreme_type : str
        The type of extreme to detect. Either "max" or "min".

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing the relative extreme values and their next values.
    """
    if data.ndim == 1:
        if extreme_type == "max":
            extrema = argrelmax(data)[0]
        elif extreme_type == "min":
            extrema = argrelmin(data)[0]
        else:
            raise ValueError("extreme_type must be either 'max' or 'min'.")

        relative_extremes = data[extrema]
        return relative_extremes[:-1], relative_extremes[1:]
    else:
        # Handle 3D arrays (N, T, D)
        results = []
        for i in range(data.shape[0]):
            for j in range(data.shape[2]):
                if extreme_type == "max":
                    extrema = argrelmax(data[i, :, j])[0]
                else:
                    extrema = argrelmin(data[i, :, j])[0]

                relative_extremes = data[i, extrema, j]
                results.append((relative_extremes[:-1], relative_extremes[1:]))

        # Stack results
        x = np.concatenate([r[0] for r in results])
        y = np.concatenate([r[1] for r in results])
        return x, y


def _poincare_section(
    data: np.ndarray, dt: float, plane: str = "x=0", direction: str = "positive"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Poincaré section of the data.

    Parameters
    ----------
    data : np.ndarray
        The input data array of shape (T,) or (N, T, D).
    dt : float
        The time step between data points.
    plane : str, optional
        The plane to use for the section. Options: "x=0", "y=0", "z=0".
    direction : str, optional
        The direction of crossing. Options: "positive", "negative".

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing the points before and after crossing the plane.
    """
    if data.ndim == 1:
        raise ValueError("Poincaré section requires at least 2D data.")

    if data.ndim == 2:
        data = data[np.newaxis, ...]

    N, T, D = data.shape

    if D < 2:
        raise ValueError("Poincaré section requires at least 2D data.")

    # Determine which dimension to use for the section
    if plane == "x=0":
        dim = 0
    elif plane == "y=0":
        dim = 1
    elif plane == "z=0" and D >= 3:
        dim = 2
    else:
        raise ValueError(f"Invalid plane: {plane}")

    crossings = []
    for i in range(N):
        # Find points where the trajectory crosses the plane
        if direction == "positive":
            mask = (data[i, :-1, dim] <= 0) & (data[i, 1:, dim] > 0)
        else:
            mask = (data[i, :-1, dim] >= 0) & (data[i, 1:, dim] < 0)

        # Get the points before and after crossing
        before = data[i, :-1][mask]
        after = data[i, 1:][mask]
        crossings.append((before, after))

    # Stack results
    before_points = np.concatenate([c[0] for c in crossings])
    after_points = np.concatenate([c[1] for c in crossings])

    return before_points, after_points


def _recurrence_plot(
    data: np.ndarray, threshold: float = 0.1, metric: str = "euclidean"
) -> np.ndarray:
    """
    Compute a recurrence plot from the data.

    Parameters
    ----------
    data : np.ndarray
        The input data array of shape (T,) or (N, T, D).
    threshold : float, optional
        The threshold for considering points as recurrent.
    metric : str, optional
        The distance metric to use. Options: "euclidean", "manhattan", "chebyshev".

    Returns
    -------
    np.ndarray
        The recurrence plot matrix.
    """
    if data.ndim == 1:
        data = data[np.newaxis, ..., np.newaxis]
    elif data.ndim == 2:
        data = data[np.newaxis, ...]

    N, T, D = data.shape

    # Compute distance matrix for each sample
    recurrence_plots = []
    for i in range(N):
        if metric == "euclidean":
            distances = np.sqrt(
                np.sum((data[i, :, np.newaxis] - data[i, np.newaxis, :]) ** 2, axis=2)
            )
        elif metric == "manhattan":
            distances = np.sum(np.abs(data[i, :, np.newaxis] - data[i, np.newaxis, :]), axis=2)
        elif metric == "chebyshev":
            distances = np.max(np.abs(data[i, :, np.newaxis] - data[i, np.newaxis, :]), axis=2)
        else:
            raise ValueError(f"Invalid metric: {metric}")

        # Create binary recurrence plot
        recurrence = distances <= threshold
        recurrence_plots.append(recurrence.astype(float))

    return np.stack(recurrence_plots)


# Plotting functions
def plot_2d(ax: plt.Axes, x: np.ndarray, y: np.ndarray, label: str, **kwargs) -> None:
    """
    Make a 2D plot on the given axis.
    """
    ax.plot(x, y, label=label, **kwargs)


def scatter_2d(ax: plt.Axes, x: np.ndarray, y: np.ndarray, label: str, **kwargs) -> None:
    """
    Make a 2D scatter plot on the given axis.
    """
    ax.scatter(x, y, label=label, **kwargs)


def plot_3d(
    ax: plt.Axes, x: np.ndarray, y: np.ndarray, z: np.ndarray, label: str, **kwargs
) -> None:
    """
    Make a 3D plot on the given axis.
    """
    ax.plot(x, y, z, label=label, **kwargs)


def scatter_3d(
    ax: plt.Axes, x: np.ndarray, y: np.ndarray, z: np.ndarray, label: str, **kwargs
) -> None:
    """
    Make a 3D scatter plot on the given axis.
    """
    ax.scatter(x, y, z, label=label, **kwargs)
