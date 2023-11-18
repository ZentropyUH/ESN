from typer import Typer
from typer import Option
from typing import List

import pandas as pd


from src.utils import load_data
from cli.enums import *


app = Typer(
    help="",
    no_args_is_help=True,
)


# FIX: Output path

@app.command(
    name="linear-single",
    no_args_is_help=True,
    help="",
)
def linear_single(
    data_file: str = Option(
        ...,
        "--data-file",
        "-df",
        help="Data to be plotted.",
    ),
    
    forecast_file: str = Option(
        None,
        "--forecast-file",
        "-ff",
        help="Forecast of the data.",
    ),
    
    transient: int = Option(
        1000,
        "--transient",
        "-tr",
        help="Transient of the data.",
    ),
    
    train_length: int = Option(
        10000,
        "--trained-length",
        "-tl",
        help="Trained length of the data.",
    ),
    
    start: int = Option(
        0,
        "--start",
        "-s",
        help="Start index of the data to be plotted.",
    ),
    
    end: int = Option(
        None,
        "--end",
        "-e",
        help="End index of the data to be plotted.",
    ),
    
    target_label: List[str] = Option(
        ["Target"],
        "--target-label",
        "-tl",
        help="Label of the target data.",
    ),
    
    forecast_label: List[str] = Option(
        ["Forecast"],
        "--forecast-label",
        "-fl",
        help="Label of the forecast data.",
    ),
    
    title: str = Option(
        "Linear Plot",
        "--title",
        "-t",
        help="Title of the plot.",
    ),
    
    delta_t: float = Option(
        1,
        "--delta-t",
        "-dt",
        help="Delta t of the data.",
    ),
    
    lyapunov_exponent: float = Option(
        1,
        "--lyapunov-exponent",
        "-le",
        help="Lyapunov exponent of the data.",
    ),
    
    xlabel: str = Option(
        "x",
        "--xlabel",
        "-xl",
        help="Label of the x axis.",
    ),
    
    filepath: str = Option(
        None,
        "--filepath",
        "-fp",
        help="Filepath of the plot.",
    ),
    
    show: bool = Option(
        False,
        "--show/--no-show",
        "-s",
        help="Show the plot.",
    ),
):
    
    from src.plots.systems import linear_single_plot
    
    if forecast_file is not None:
        
        if transient != 0 or train_length != 0:
            (
                _,
                _,
                _,
                _,
                _,
                data
            ) = load_data(data_file, transient, train_length)
        else:
            data = pd.read_csv(data_file, header=None).to_numpy()

        forecast = pd.read_csv(forecast_file, header=None).to_numpy()

    else:
        data = pd.read_csv(data_file, header=None).to_numpy()
        forecast = None
    
    if len(target_label) == 1:
        target_label = target_label[0]
    if len(forecast_label) == 1:
        forecast_label = forecast_label[0]
    
    linear_single_plot(
        target=data,
        forecast=forecast,

        start=start,
        end=end,

        target_labels=target_label,
        forecast_labels=forecast_label,
        title=title,
        dt=delta_t,
        lyapunov_exponent=lyapunov_exponent,
        xlabel=xlabel,
        filepath=filepath,
        show=show,
    )


@app.command(
    name="linear-multi",
    no_args_is_help=True,
    help="",
)
def linear_multi(
    data_file: str = Option(
        ...,
        "--data-file",
        "-df",
        help="Data to be plotted.",
    ),
    
    forecast_file: str = Option(
        None,
        "--forecast-file",
        "-ff",
        help="Forecast of the data.",
    ),
    
    transient: int = Option(
        1000,
        "--transient",
        "-tr",
        help="Transient of the data.",
    ),
    
    train_length: int = Option(
        10000,
        "--trained-length",
        "-tl",
        help="Trained length of the data.",
    ),
    
    start: int = Option(
        0,
        "--start",
        "-s",
        help="Start index of the data to be plotted.",
    ),
    
    end: int = Option(
        None,
        "--end",
        "-e",
        help="End index of the data to be plotted.",
    ),
    
    target_label: List[str] = Option(
        ["Target"],
        "--target-label",
        "-tl",
        help="Label of the target data.",
    ),
    
    forecast_label: List[str] = Option(
        ["Forecast"],
        "--forecast-label",
        "-fl",
        help="Label of the forecast data.",
    ),
    
    title: str = Option(
        "Linear Plot",
        "--title",
        "-t",
        help="Title of the plot.",
    ),
    
    delta_t: float = Option(
        1,
        "--delta-t",
        "-dt",
        help="Delta t of the data.",
    ),
    
    lyapunov_exponent: float = Option(
        1,
        "--lyapunov-exponent",
        "-le",
        help="Lyapunov exponent of the data.",
    ),
    
    xlabel: str = Option(
        "x",
        "--xlabel",
        "-xl",
        help="Label of the x axis.",
    ),
    
    filepath: str = Option(
        None,
        "--filepath",
        "-fp",
        help="Filepath of the plot.",
    ),
    
    show: bool = Option(
        False,
        "--show/--no-show",
        "-s",
        help="Show the plot.",
    ),
):
    if len(target_label) == 1:
        target_label = target_label[0]
    if len(forecast_label) == 1:
        forecast_label = forecast_label[0]
    
    from src.plots.systems import linear_multiplot
    
    if forecast_file is not None:
        
        if transient != 0 or train_length != 0:
            (
                _,
                _,
                _,
                _,
                _,
                data
            ) = load_data(data_file, transient, train_length)
        else:
            data = pd.read_csv(data_file, header=None).to_numpy()
            
        forecast = pd.read_csv(forecast_file, header=None).to_numpy()
    
    else:
        data = pd.read_csv(data_file, header=None).to_numpy()
        forecast = None
       

    
     
    linear_multiplot(
        target=data,
        forecast=forecast,
        
        start=start,
        end=end,
        
        target_labels=target_label,
        forecast_labels=forecast_label,
        title=title,
        dt=delta_t,
        lyapunov_exponent=lyapunov_exponent,
        xlabel=xlabel,
        filepath=filepath,
        show=show,
    )


@app.command(
    name="contourf",
    no_args_is_help=True,
    help="",
)
def contourf(
    data_file: str = Option(
        ...,
        "--data-file",
        "-df",
        help="Data to be plotted.",
    ),
    
    forecast_file: str = Option(
        None,
        "--forecast-file",
        "-ff",
        help="Forecast of the data.",
    ),
    
    transient: int = Option(
        1000,
        "--transient",
        "-tr",
        help="Transient of the data.",
    ),
    
    train_length: int = Option(
        10000,
        "--train-length",
        "-tl",
        help="Trained length of the data.",
    ),
    
    start: int = Option(
        0,
        "--start",
        "-s",
        help="Start index of the data to be plotted.",
    ),
    
    end: int = Option(
        None,
        "--end",
        "-e",
        help="End index of the data to be plotted.",
    ),
    
    title: str = Option(
        "Contourf Plot",
        "--title",
        "-t",
        help="Title of the plot.",
    ),
    
    delta_t: float = Option(
        1,
        "--delta-t",
        "-dt",
        help="Delta t of the data.",
    ),
    
    lyapunov_exponent: float = Option(
        1,
        "--lyapunov-exponent",
        "-le",
        help="Lyapunov exponent of the data.",
    ),
    
    renorm_y: float = Option(
        None,
        "--renorm-y",
        "-ry",
        help="Renormalization of the y axis.",
    ),
    
    xlabel: str = Option(
        r'$\Lambda t$',
        "--xlabel",
        "-xl",
        help="Label of the x axis.",
    ),
    
    filepath: str = Option(
        None,
        "--filepath",
        "-fp",
        help="Filepath of the plot.",
    ),
    
    show: bool = Option(
        False,
        "--show/--no-show",
        "-s",
        help="Show the plot.",
    ),
    
    target_label: str = Option(
        "Original",
        "--target-label",
        "-tl",
        help="Label of the target data.",
    ),
    
    forecast_label: str = Option(
        "Forecast",
        "--forecast-label",
        "-fl",
        help="Label of the forecast data.",
    ),
    cmap: str = Option(
        "viridis",
        "--cmap",
        "-cm",
        help="Colormap of the plot.",
    ),
):
    
    from src.plots.systems import contourf_plot
    
    if forecast_file is not None:
        
        if transient != 0 or train_length != 0:
            (
                _,
                _,
                _,
                _,
                _,
                data
            ) = load_data(data_file, transient, train_length)
        else:
            data = pd.read_csv(data_file, header=None).to_numpy()
            
        forecast = pd.read_csv(forecast_file, header=None).to_numpy()
        
    
    else:
        data = pd.read_csv(data_file, header=None).to_numpy()
        forecast = None
        
    contourf_plot(
        target=data,
        forecast=forecast,
        
        start=start,
        end=end,
        
        title=title,
        dt=delta_t,
        lyapunov_exponent=lyapunov_exponent,
        renorm_y=renorm_y,
        xlabel=xlabel,
        filepath=filepath,
        show=show,
        target_label=target_label,
        forecast_label=forecast_label,
        cmap=cmap,
    )


@app.command(
    name="3d",
    no_args_is_help=True,
    help="",
)
def plot_3D(
    data_file: str = Option(
        ...,
        "--data-file",
        "-df",
        help="Data to be plotted.",
    ),
    
    forecast_file: str = Option(
        None,
        "--forecast-file",
        "-ff",
        help="Forecast of the data.",
    ),
    
    transient: int = Option(
        1000,
        "--transient",
        "-tr",
        help="Transient of the data.",
    ),
    
    train_length: int = Option(
        10000,
        "--train-length",
        "-tl",
        help="Trained length of the data.",
    ),
    
    start: int = Option(
        0,
        "--start",
        "-s",
        help="Start index of the data to be plotted.",
    ),
    
    end: int = Option(
        None,
        "--end",
        "-e",
        help="End index of the data to be plotted.",
    ),
    
    title: str = Option(
        "3D Plot",
        "--title",
        "-t",
        help="Title of the plot.",
    ),
    
    target_label: str = Option(
        "Original",
        "--target-label",
        "-tl",
        help="Label of the target data.",
    ),
    
    forecast_label: str = Option(
        "Forecast",
        "--forecast-label",
        "-fl",
        help="Label of the forecast data.",
    ),
    
    xlabels: List[str] = Option(
        ['x'],
        "--xlabel",
        "-xl",
        help="Label of the x axis.",
    ),
    
    ylabels: List[str] = Option(
        ['y'],
        "--ylabel",
        "-yl",
        help="Label of the y axis.",
    ),
    
    zlabels: List[str] = Option(
        ['z'],
        "--zlabel",
        "-zl",
        help="Label of the z axis.",
    ),
    
    filepath: str = Option(
        None,
        "--filepath",
        "-fp",
        help="Filepath of the plot.",
    ),
    
    show: bool = Option(
        False,
        "--show/--no-show",
        "-s",
        help="Show the plot.",
    ),
    
    single_plot: bool = Option(
        True,
        "--single-plot/--separate-plot",
        "-sp",
        help="Show the plot.",
    ),
    
):
    if len(xlabels) == 1:
        xlabels = xlabels[0]
    if len(ylabels) == 1:
        ylabels = ylabels[0]
    if len(zlabels) == 1:
        zlabels = zlabels[0]
    
    from src.plots.systems import plot3D
    
    if forecast_file is not None:
        if transient != 0 or train_length != 0:
            (
                _,
                _,
                _,
                _,
                _,
                data
            ) = load_data(data_file, transient, train_length)
        else:
            data = pd.read_csv(data_file, header=None).to_numpy()
            
        forecast = pd.read_csv(forecast_file, header=None).to_numpy()
        
    
    else:
        data = pd.read_csv(data_file, header=None).to_numpy()
        forecast = None
        
    data = pd.read_csv(data_file, header=None).to_numpy()
    
    
    plot3D(
        target=data,
        forecast=forecast,
        start=start,
        end=end,
        target_label=target_label,
        forecast_label=forecast_label,
        xlabels=xlabels,
        ylabels=ylabels,
        zlabels=zlabels,
        title=title,
        filepath=filepath,
        single_plot=single_plot,
        show=show,
    )

@app.command(
    name="max-return",
    no_args_is_help=True,
    help="",
)
def max_return(
    data_file: str = Option(
        ...,
        "--data-file",
        "-df",
        help="Data to be plotted.",
    ),
    
    forecast_file: str = Option(
        None,
        "--forecast-file",
        "-ff",
        help="Forecast of the data.",
    ),
    
    target_label: List[str] = Option(
        ["Original"],
        "--target-label",
        "-tl",
        help="Label of the target data.",
    ),
    
    forecast_label: List[str] = Option(
        ["Forecast"],
        "--forecast-label",
        "-fl",
        help="Label of the forecast data.",
    ),
    
            
    title: str = Option(
        "Max Return Plot",
        "--title",
        "-t",
        help="Title of the plot.",
    ),
    
    filepath: str = Option(
        None,
        "--filepath",
        "-fp",
        help="Filepath of the plot.",
    ),
    
    show: bool = Option(
        False,
        "--show/--no-show",
        "-s",
        help="Show the plot.",
    ),
):
    if len(target_label) == 1:
        target_label = target_label[0]
    if len(forecast_label) == 1:
        forecast_label = forecast_label[0]
    
    from src.plots.systems import max_return_map
    
    if forecast_file is not None:
        forecast = pd.read_csv(forecast_file, header=None).to_numpy()
    else:
        forecast = None

    data = pd.read_csv(data_file, header=None).to_numpy()

        
    max_return_map(
        target=data,
        forecast=forecast,
        target_labels=target_label,
        forecast_labels=forecast_label,
        title=title,
        filepath=filepath,
        show=show,
    )

if __name__ == "__main__":
    app()
