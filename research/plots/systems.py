from typing import Iterable, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import find_peaks

from src.utils import letter

#TODO: Parameter control and assertions
#TODO: Change _base_plot label and target_label to target_label and forecast_label, means refactoring
#TODO: make possible to choose letter or number for the labels
#TODO: Make it so that if the data is too long in the linear plots, the screenview starts moving right when data reaches the middle and continues moving right until the end of the data with the same speed (Hard one )
#TODO: Check everything is working as expected

def _base_setup_plot(
    features: int,
    cols: int = 1,
    title: str = '',
    xlabel: str = '',
    figsize: Tuple[float, float] = (20, 9.6),
    sharey: bool = True,
) -> Union[Tuple[Figure, Axes], Tuple[Figure, List[Axes]]]:
    fig, axs = plt.subplots(features, cols, sharey=sharey, figsize=figsize)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9, bottom=0.08, hspace=0.3)
    fig.suptitle(title, fontsize=16)
    fig.supxlabel(xlabel)
    if features == 1:
        axs = [axs]
    return fig, axs

def _base_setup_plot_3D(
    single_plot: bool = True,
    title: str = '',
    figsize: Tuple[float, float] = (20, 9.6),
) -> Union[Tuple[Figure, Axes], Tuple[Figure, List[Axes]]]:
    
    if single_plot:
        fig, axs = plt.subplots(1, 1, subplot_kw={'projection': '3d'}, figsize=figsize)
        axs = [axs]
    else:
        fig, axs = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=figsize)
        
        def on_move(event):
            """ Event handler to synchronize rotation of two 3D plots. """
            if event.inaxes == axs[0]:
                # Sync the second plot to the first
                axs[1].view_init(elev=axs[0].elev, azim=axs[0].azim)
            elif event.inaxes == axs[1]:
                # Sync the first plot to the second
                axs[0].view_init(elev=axs[1].elev, azim=axs[1].azim)
            fig.canvas.draw_idle()
        
        fig.canvas.mpl_connect('motion_notify_event', on_move)
        
    fig.tight_layout()
    fig.subplots_adjust(top=0.9, bottom=0.08, hspace=0.3)
    fig.suptitle(title, fontsize=16)
    return fig, axs

def _base_plot(
    ax: Axes,
    xvalues: np.ndarray,
    val_target: np.ndarray,
    forecast: np.ndarray = None,
    label="target",
    target_label: str = '',
) -> None:
    ax.plot(xvalues, val_target, label=label)
    if forecast is not None:
        ax.plot(xvalues, forecast, label=target_label, linestyle='--')
    ax.legend()
    
def _base_scatter(
    ax: Axes,
    xvalues: np.ndarray,
    val_target: np.ndarray,
    fxvalues: np.ndarray = None,
    forecast: np.ndarray = None,
    label: str ="target",
    target_label: str = '',
    size : int = 1
) -> None:
    ax.scatter(xvalues, val_target, label=label, s=size)
    if forecast is not None:
        ax.scatter(fxvalues, forecast, label=target_label, marker='x', s=size)
    ax.legend()

def _base_plot_3D(
    ax: Axes,
    target: np.ndarray,
    forecast: np.ndarray = None,
    target_label : str = '',
    forecast_label : str = '',
    xlabels: Union[str, Iterable[str]] = 'x',
    ylabels: Union[str, Iterable[str]] = 'y',
    zlabels: Union[str, Iterable[str]] = 'z',
    color: str = None,
) -> None:
    
    # Handling different types of labels
    if isinstance(xlabels, str):
        xlabels = [''.join([xlabels, f'_{i}']) for i in range(2)]
    if isinstance(ylabels, str):
        ylabels = [''.join([ylabels, f'_{i}']) for i in range(2)]
    if isinstance(zlabels, str):
        zlabels = [''.join([zlabels, f'_{i}']) for i in range(2)]
    
    ax.plot(target[:, 0], target[:, 1], target[:, 2], label=target_label, lw=1.5, color=color)
    
    if forecast is not None:
        ax.plot(forecast[:, 0], forecast[:, 1], forecast[:, 2], label=forecast_label, linestyle='--', lw=0.8)
        
    ax.set_xlabel(xlabels[0])
    ax.set_ylabel(ylabels[0])
    ax.set_zlabel(zlabels[0])
    ax.legend()

def _contourf_plot(
    ax: Axes,
    xvalues: np.ndarray,
    yvalues: np.ndarray,
    data: np.ndarray,
    levels: int,
    title: str = '',
    cmap: str = 'viridis'
) -> None:
    plot = ax.contourf(xvalues, yvalues, data.T, levels=levels, cmap=cmap)
    ax.set_title(title)
    return plot

def plot_forecast(
    forecast: np.ndarray,
    val_target: np.ndarray,
    dt: float = 1,
    lyapunov_exponent: float = 1,
    title: str = '',
    xlabel: str = 't',
    cmap: str = 'viridis',
    filepath: str = None,
    show: bool = False,
) -> None:
    '''Plots the prediction and the target values.'''

    # initial data
    dt = dt * lyapunov_exponent
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
            ax=axs[0],
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
        error = forecast - val_target

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
            levels=50,
            title='Original model',
            cmap=cmap,
        )
        prediction_plot = _contourf_plot(
            ax=axs[1],
            xvalues=xvalues,
            yvalues=yvalues,
            data=forecast,
            levels=50,
            title='Predicted model',
            cmap=cmap,
        )
        error_plot = _contourf_plot(
            ax=axs[2],
            xvalues=xvalues,
            yvalues=yvalues,
            data=error,
            levels=50,
            title='Error',
            cmap=cmap,
        )

        # Individually adding the colorbars
        fig.colorbar(model_plot, ax=axs[0], cmap=cmap)
        fig.colorbar(prediction_plot, ax=axs[1], cmap=cmap)
        fig.colorbar(error_plot, ax=axs[2], cmap=cmap)
    
    if filepath:
        plt.savefig(filepath)
    if show:
        plt.show()

def plot_system(
    target,
    length: int,
    title: str = '',
    dt: float = 1,
    lyapunov_exponent: float = 1,
    xlabel: str = 't',
    cmap: str = 'viridis',
    filepath: str = None,
    show: bool = False,
) -> None:
    dt = dt * lyapunov_exponent
    features = target.shape[-1]
    _target = target[0][:length]
    xvalues = np.arange(0, length) * dt
    yvalues = np.arange(0, target.shape[-1])

    if features == 1:
        fig, axs = _base_setup_plot(
            features=features,
            title=title,
            xlabel=xlabel,
            figsize=(16, 6.6)
        )
        _base_plot(
            ax=axs[0],
            xvalues=xvalues,
            val_target=_target,
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
                val_target=_target[:, i],
            )
    
    else:
        fig, axs = _base_setup_plot(
            features=1,
            title=title,
            xlabel=xlabel,
            figsize=(16, 4.6)
        )
        model_plot = _contourf_plot(
            ax=axs,
            xvalues=xvalues,
            yvalues=yvalues,
            data=_target,
            levels=50,
            cmap=cmap,
        )
        fig.colorbar(model_plot, ax=axs, cmap=cmap)

    if filepath:
        plt.savefig(filepath)
    if show:
        plt.show()


#### NEW GENERATION PLOTS ####

def animate_drawing(
    fig: Figure, 
    axs: Union[Axes, List[Axes]],
    interval: int = 50
) -> FuncAnimation:
    axs = np.array(axs).flatten()

    lines_2d, lines_3d, line_data_2d, line_data_3d = [], [], [], []
    markers_2d, markers_3d = [], []

    # Separate 2D and 3D lines and create markers
    for ax in axs:
        if isinstance(ax, Axes3D):
            for line in ax.lines:
                lines_3d.append(line)
                x, y, z = line.get_data_3d()
                line_data_3d.append((x, y, z))
                markers_3d.append(ax.plot(x[0], y[0], z[0], marker='o', color='red', markersize=5)[0])
        else:
            for line in ax.lines:
                lines_2d.append(line)
                x, y = line.get_data()
                line_data_2d.append((x, y))
                markers_2d.append(ax.plot(x[0], y[0], marker='o', color='red', markersize=5)[0])

    def init():
        for line in lines_2d:
            line.set_data([], [])
        for marker in markers_2d:
            marker.set_data([], [])
        for line in lines_3d:
            line.set_data_3d([], [], [])
        for marker in markers_3d:
            marker.set_data_3d([], [], [])
        return lines_2d + lines_3d + markers_2d + markers_3d

    def animate(i):
        for j, line in enumerate(lines_2d):
            x, y = line_data_2d[j]
            line.set_data(x[:i], y[:i])
            markers_2d[j].set_data(x[i-1:i], y[i-1:i])
        for j, line in enumerate(lines_3d):
            x, y, z = line_data_3d[j]
            line.set_data_3d(x[:i], y[:i], z[:i])
            markers_3d[j].set_data_3d(x[i-1:i], y[i-1:i], z[i-1:i])
        return lines_2d + lines_3d + markers_2d + markers_3d

    max_frames_2d = len(max(line_data_2d, key=lambda x: len(x[0]))[0]) if line_data_2d else 0
    max_frames_3d = len(max(line_data_3d, key=lambda x: len(x[0]))[0]) if line_data_3d else 0
    max_frames = max(max_frames_2d, max_frames_3d)

    anim = FuncAnimation(fig, animate, init_func=init, frames=max_frames, interval=interval, blit=True)
    return anim


def linear_single_plot(
    target: np.ndarray,
    forecast: np.ndarray = None,
    
    start=0,
    end=None,
    
    target_labels: Union[str, Iterable[str]] = 'system',
    forecast_labels: Union[str, Iterable[str]] = 'forecast',
    title: str = '',
    dt: float = 1,
    lyapunov_exponent: float = 1,
    xlabel: str = r'\Lambda t',
    filepath: str = None,
    show: bool = False,
    animate: bool = False,
    frame_interval: int = 20,
) -> None:
    
    assert start >= 0, 'Start index must be greater or equal to 0.'
    assert end is None or end > start, 'End index must be greater than start index.'

    dt = dt * lyapunov_exponent
    features = target.shape[-1]
    
    # Handling different types of labels
    if isinstance(target_labels, str):
        target_labels = [''.join([target_labels, f'_{i}']) for i in range(features)]
    if isinstance(forecast_labels, str) and forecast is not None:
        forecast_labels = [''.join([forecast_labels, f'_{i}']) for i in range(features)]
    
    if forecast is not None:
        length = min(forecast.shape[0], target.shape[0])
        forecast = forecast[:length]
        target = target[:length]
    
    if end is None:
        end = target.shape[0]
    
    length = end - start
    
    target = target[start:end]
    
    if forecast is not None:
        forecast = forecast[start:end]
    
    
    
    xvalues = np.arange(start, end) * dt * lyapunov_exponent
    
    fig, axs = _base_setup_plot(
        features=1,
        title=title,
        xlabel=xlabel,
        figsize=(16, 6)
    )
    
    
    for i in range(features):
        _base_plot(
            ax=axs[0],
            xvalues=xvalues,
            val_target=target[:, i],
            forecast=forecast[:, i] if forecast is not None else None,
            label=target_labels[i],
            target_label=forecast_labels[i] if forecast is not None else None,
        )
    
    if animate:
        anim = animate_drawing(fig, axs, interval=frame_interval)
        
        if filepath:
            anim.save(filepath, writer='ffmpeg')
        
        if show:
            plt.show()
    else:
        
        if filepath:
            plt.savefig(filepath)
        
        if show:
            plt.show()

def linear_multiplot(
    target: np.ndarray,
    forecast: np.ndarray = None,
    start = 0,
    end = None,
    target_labels: Union[str, Iterable[str]] = 'system',
    forecast_labels: Union[str, Iterable[str]] = 'forecast',
    title: str = '',
    dt: float = 1,
    lyapunov_exponent: float = 1,
    xlabel: str = r'$\Lambda t$',
    filepath: str = None,
    show: bool = False,
    animate: bool = False,
    frame_interval: int = 20,
) -> None:
    
    assert start >= 0, 'Start index must be greater or equal to 0.'
    assert end is None or end > start, 'End index must be greater than start index.'

    dt = dt * lyapunov_exponent
    features = target.shape[-1]
    
    # Handling different types of labels
    if isinstance(target_labels, str):
        target_labels = [''.join([target_labels, f'_{i}']) for i in range(features)]
    if isinstance(forecast_labels, str) and forecast is not None:
        forecast_labels = [''.join([forecast_labels, f'_{i}']) for i in range(features)]
        
    
    if end is None:
        end = target.shape[0]
        
    target = target[start:end]
    
    if forecast is not None:
        forecast = forecast[start:end]
    
    xvalues = np.arange(start, end) * dt * lyapunov_exponent
    
    fig, axs = _base_setup_plot(
        features=features,
        title=title,
        xlabel=xlabel,
        figsize=(16, 6)
    )
    
    for i in range(features):
        _base_plot(
            ax=axs[i],
            xvalues=xvalues,
            val_target=target[:, i],
            forecast=forecast[:, i] if forecast is not None else None,
            label=target_labels[i],
            target_label=forecast_labels[i] if forecast is not None else None,
        )
    
    if animate:
        anim = animate_drawing(fig, axs, interval=frame_interval)
        
        if filepath:
            anim.save(filepath, writer='ffmpeg')
        
        if show:
            plt.show()
    else:
        
        if filepath:
            plt.savefig(filepath)

        if show:
            plt.show()

def contourf_plot(
    target: np.ndarray,
    forecast: np.ndarray = None,
    start = 0,
    end = None,
    title: str = '',
    dt: float = 1,
    lyapunov_exponent: float = 1,
    renorm_y: float = None,
    xlabel: str = r'$\Lambda t$',
    filepath: str = None,
    show: bool = False,
    target_label: str = '',
    forecast_label: str = '',
    cmap: str = 'viridis',
) -> None:
    
    assert start >= 0, 'Start index must be greater or equal to 0.'
    assert end is None or end > start, 'End index must be greater than start index.'
    
    dt = dt * lyapunov_exponent
    features = target.shape[-1]

    if forecast is not None:
        length = min(forecast.shape[0], target.shape[0])
        forecast = forecast[:length]
        target = target[:length]
        fig, axs = _base_setup_plot(features=3, title=title, xlabel=xlabel)
    else:
        fig, axs = _base_setup_plot(features=1, title=title, xlabel=xlabel)

    
    if end is None:
        end = target.shape[0]
    
    length = end - start
        
    target = target[start:end]
        
    if forecast is not None:
        forecast = forecast[start:end]
        error = forecast - target
    
    
    xvalues = np.arange(start, end) * dt * lyapunov_exponent
    yvalues = np.arange(0, features)

    if renorm_y is not None:
        yvalues *= renorm_y

    # Plot for target
    model_plot = _contourf_plot(
        ax=axs[0],
        xvalues=xvalues,
        yvalues=yvalues,
        data=target,
        levels=50,
        title=target_label,
        cmap=cmap,
    )

    fig.colorbar(model_plot, ax=axs[0], cmap=cmap)
    
    if forecast is not None:
        # Plot for forecast
        prediction_plot = _contourf_plot(
            ax=axs[1],
            xvalues=xvalues,
            yvalues=yvalues,
            data=forecast,
            levels=50,
            title=forecast_label,
            cmap=cmap,
        )

        # Plot for error
        error_plot = _contourf_plot(
            ax=axs[2],
            xvalues=xvalues,
            yvalues=yvalues,
            data=error,
            levels=50,
            title='Error',
            cmap=cmap,
        )
    
        fig.colorbar(prediction_plot, ax=axs[1], cmap=cmap)
        fig.colorbar(error_plot, ax=axs[2], cmap=cmap)
    
    
    if filepath:
        plt.savefig(filepath)

    if show:
        plt.show()
    
def plot3D(
    target: np.ndarray,
    forecast: np.ndarray = None,
    
    start=0,
    end=None,
    
    target_label: str = 'system',
    forecast_label: str = 'forecast',
    
    xlabels: Union[str, Iterable[str]] = 'x',
    ylabels: Union[str, Iterable[str]] = 'y',
    zlabels: Union[str, Iterable[str]] = 'z',

    title: str = '',
    
    filepath: str = None,
    single_plot: bool = True,
    show: bool = False,
    animate: bool = False,
    frame_interval: int = 20,
) -> None:
    
    assert start >= 0, 'Start index must be greater or equal to 0.'
    assert end is None or end > start, 'End index must be greater than start index.'
    
    # Handling different types of labels
    if isinstance(xlabels, str):
        xlabels = [''.join([xlabels, f'_{i}']) for i in range(2)]
    if isinstance(ylabels, str):
        ylabels = [''.join([ylabels, f'_{i}']) for i in range(2)]
    if isinstance(zlabels, str):
        zlabels = [''.join([zlabels, f'_{i}']) for i in range(2)]
        
    if end is None:
        end = target.shape[0]
        
    target = target[start:end]
    
    if forecast is not None:
        forecast = forecast[start:end]
    
    fig, axs = _base_setup_plot_3D(
        single_plot=single_plot,
        title=title,
        figsize=None
    )   
    
    if single_plot:
        _base_plot_3D(
            ax=axs[0],
            target=target,
            forecast=forecast,
            target_label=target_label,
            forecast_label=forecast_label,
            xlabels=xlabels[0],
            ylabels=ylabels[0],
            zlabels=zlabels[0],
        )
    else:
        _base_plot_3D(
            ax=axs[0],
            target=target,
            # target_label=target_label,
            xlabels=xlabels[0],
            ylabels=ylabels[0],
            zlabels=zlabels[0],
        )
        
        axs[0].set_title(target_label)
        
        _base_plot_3D(
            ax=axs[1],
            target=forecast,
            # target_label=forecast_label,
            xlabels=xlabels[1],
            ylabels=ylabels[1],
            zlabels=zlabels[1],
            color='orange'
        )
        
        axs[1].set_title(forecast_label)
    
    if animate:
        anim = animate_drawing(fig, axs, interval=frame_interval)
        
        if filepath:
            anim.save(filepath, writer='ffmpeg')
        
        if show:
            plt.show()
    
    else:    
        if filepath:
            plt.savefig(filepath)
        
        if show:
            plt.show()

def max_return_map(
    target: np.ndarray,
    forecast: np.ndarray = None,
    
    target_labels: Union[str, Iterable[str]] = 'system',
    forecast_labels: Union[str, Iterable[str]] = 'forecast',
    
    title: str = '',
    filepath: str = None,
    show: bool = False,  
) -> None:
    
    features = target.shape[-1]
    
    
    
    if isinstance(target_labels, str):
        target_labels = [''.join([target_labels, f'_{letter(i)}']) for i in range(features)]
    if isinstance(forecast_labels, str) and forecast is not None:
        forecast_labels = [''.join([forecast_labels, f'_{letter(i)}']) for i in range(features)]
    
    fig, axs = _base_setup_plot(
        features=1,
        cols=features,
        title=title,
        figsize=(18, 6),
        sharey=False,
    )
        
    
    for i in range(features):
        
        dimension = target[:, -(i+1)]
        peaks, _ = find_peaks(dimension)
        maxima = dimension[peaks]
        
        
        if forecast is not None:
            fdimension = forecast[:, -(i+1)]
            fpeaks, _ = find_peaks(fdimension)
            fmaxima = fdimension[fpeaks]
        
                
        _base_scatter(
            ax=axs[0][i],
            xvalues=maxima[:-1],
            val_target=maxima[1:],
            fxvalues=fmaxima[:-1] if forecast is not None else None,
            forecast=fmaxima[1:] if forecast is not None else None,
            label=target_labels[i],
            target_label=forecast_labels[i],
        )
        
        
            
    
    if filepath:
        plt.savefig(filepath)
        
    if show:
        plt.show()
    
def min_return_map(
    target: np.ndarray,
    forecast: np.ndarray = None,
    
    target_labels: Union[str, Iterable[str]] = 'system',
    forecast_labels: Union[str, Iterable[str]] = 'forecast',
    
    title: str = '',
    filepath: str = None,
    show: bool = False,  
) -> None:
    
    features = target.shape[-1]
    
    
    
    if isinstance(target_labels, str):
        target_labels = [''.join([target_labels, f'_{letter(i)}']) for i in range(features)]
    if isinstance(forecast_labels, str) and forecast is not None:
        forecast_labels = [''.join([forecast_labels, f'_{letter(i)}']) for i in range(features)]
    
    fig, axs = _base_setup_plot(
        features=1,
        cols=features,
        title=title,
        figsize=(18, 6),
        sharey=False,
    )
        
    
    for i in range(features):
        
        dimension = target[:, -(i+1)]
        peaks, _ = find_peaks(-dimension)
        maxima = dimension[peaks]
        
        
        if forecast is not None:
            fdimension = forecast[:, -(i+1)]
            fpeaks, _ = find_peaks(-fdimension)
            fmaxima = fdimension[fpeaks]
        
                
        _base_scatter(
            ax=axs[0][i],
            xvalues=maxima[:-1],
            val_target=maxima[1:],
            fxvalues=fmaxima[:-1] if forecast is not None else None,
            forecast=fmaxima[1:] if forecast is not None else None,
            label=target_labels[i],
            target_label=forecast_labels[i],
        )
        
        
            
    
    if filepath:
        plt.savefig(filepath)
        
    if show:
        plt.show()