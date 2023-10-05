from typing import List
from typing import Union
from typing import Optional
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def _base_setup_plot(
    features: int,
    title: str = '',
    xlabel: str = 't',
    figsize: Tuple[float, float] = (20, 9.6),
) -> Union[Tuple[Figure, Axes], Tuple[Figure, List[Axes]]]:
    fig, axs = plt.subplots(features, 1, sharey=True, figsize=figsize)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9, bottom=0.08, hspace=0.3)
    fig.suptitle(title, fontsize=16)
    fig.supxlabel(xlabel)
    return fig, axs


def _base_plot(
    ax: Axes,
    xvalues: np.ndarray,
    val_target: np.ndarray,
    forecast: np.ndarray = None,
):
    ax.plot(xvalues, val_target, label="target")
    if forecast is not None:
        ax.plot(xvalues, forecast, label="forecast", linestyle='--')
        ax.legend()


def _contourf_plot(
    ax: Axes,
    xvalues: np.ndarray,
    yvalues: np.ndarray,
    data: np.ndarray,
    levels: int,
    title: str = '',
    cmap: str = 'viridis'
):
    ax.contourf(xvalues, yvalues, data.T, levels=levels, cmap=cmap)
    ax.set_title(title)


def plot_forecast(
    forecast: np.ndarray,
    val_target: np.ndarray,
    dt: float = 1,
    title: str = '',
    xlabel: str = 't',
    cmap: str = 'viridis'
):
    '''Plots the prediction and the target values.'''

    # initial data
    forecast_length = forecast.shape[0]
    features = forecast.shape[-1]
    xvalues = np.arange(0, forecast_length) * dt
    yvalues = np.arange(0, forecast.shape[-1])

    if features == 1:
        fig, axs = _base_setup_plot(
            features=features,
            title=title,
            xlabel=xlabel
        )
        _base_plot(
            ax=axs,
            xvalues=xvalues,
            val_target=val_target[:, 0],
            forecast=forecast[:, 0],
        )
    elif features <= 3:
        fig, axs = _base_setup_plot(
            features=features,
            title=title,
            xlabel=xlabel
        )
        for i in range(features):
            _base_plot(
                ax=axs[i],
                xvalues=xvalues,
                val_target=val_target[:, i],
                forecast=forecast[:, i],
            )
    else:
        # calculate error between forecast and target
        error = abs(forecast - val_target)

        fig, axs = _base_setup_plot(
            features=3,
            title=title,
            xlabel=xlabel
        )
        model_plot = _contourf_plot(
            ax=axs[0],
            xvalues=xvalues,
            yvalues=yvalues,
            data=val_target,
            levels=features,
            title='Original model',
            cmap=cmap,
        )
        prediction_plot = _contourf_plot(
            ax=axs[1],
            xvalues=xvalues,
            yvalues=yvalues,
            data=forecast,
            levels=features,
            title='Predicted model',
            cmap=cmap,
        )
        error_plot = _contourf_plot(
            ax=axs[2],
            xvalues=xvalues,
            yvalues=yvalues,
            data=error,
            levels=features,
            title='Error',
            cmap=cmap,
        )

        # Individually adding the colorbars
        fig.colorbar(model_plot, ax=axs[0])
        fig.colorbar(prediction_plot, ax=axs[1])
        fig.colorbar(error_plot, ax=axs[2])
    
    plt.show()


def plot_system(
    target,
    length: int,
    title: str,
    dt: float=1,
    xlabel: str = 't',
    cmap: str = 'viridis'
):
    features = target.shape[-1]
    _target = target[:, :length, :]
    xvalues = np.arange(0, length) * dt

    if features == 1:
        fig, axs = _base_setup_plot(
            features=features,
            title=title,
            xlabel=xlabel,
            figsize=(16, 6.6)
        )
        _base_plot(
            ax=axs,
            xvalues=xvalues,
            val_target=_target[0, :],
        )
    
    elif features <= 3:
        fig, axs = _base_setup_plot(
            features=features,
            title=title,
            xlabel=xlabel,
        )
        for i in range(features):
            _base_plot(
                ax=axs[i],
                xvalues=xvalues,
                val_target=_target[0, :, i],
            )
    
    else:

        _target = target[:, :length, :][0]
        xvalues = np.arange(0, length) * dt
        yvalues = np.arange(0, target.shape[-1])
        
        fig, axs = _base_setup_plot(
            features=1,
            title=title,
            xlabel=xlabel
        )
        model_plot = _contourf_plot(
            ax=axs,
            xvalues=xvalues,
            yvalues=yvalues,
            data=_target,
            levels=features,
            cmap=cmap,
        )
        fig.colorbar(model_plot, ax=axs)

    plt.show()
