from typer import Typer
from typer import Option

from cli.enums import *


app = Typer(
    help="",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


# FIX: Output path
@app.command(
    name="plot",
    no_args_is_help=True,
    help="",
)
def plot_ESN(
    plot_type: EnumPlotType = Option(
        "linear",
        "--plot-type",
        "-pt",
        help="The type of plot to be made. The default is linear.",
    ),
    predictions: str = Option(
        ...,
        "--predictions",
        "-pr",
        help="The path to the file containing the predictions.",
    ),
    data_file: str = Option(
        ...,
        "--data_file",
        "-df",
        help="The path to the file containing the data.",
    ),
    lyapunov_exponent: float = Option(
        1,
        "--lyapunov-exponent",
        "-le",
        help="The lyapunov exponent of the data. The default is 1. It is used to scale the time to the lyapunov time units.",
    ),
    delta_time: float = Option(
        1,
        "--delta-time",
        "-dt",
        help="The time step between each point in the data. The default is 1.",
    ),
    plot_points: int = Option(
        None,
        "--plot-points",
        "-pp",
        help="The number of points/steps to plot.",
    ),
    title: str = Option(
        "",
        "--title",
        "-T",
        help="The title of the plot. The default is empty.",
    ),
    save_path: str = Option(
        ".",
        "--save-path",
        "-sp",
        help="The path where the plot will be saved. The default is None.",
    ),
    show: bool = Option(
        False,
        "--show/--no-show",
        "-s/-ns",
        help="Whether to show the plot or not. The default is True.",
    ),
    y_labels: str = Option(
        None,
        "--y-labels",
        "-yl",
        help="The labels of the y axis. The default is None. Value should be a string with the labels separated by commas.",
    ),
    y_values: str = Option(
        None,
        "--y-values",
        "-yv",
        help="The values of the y axis. The default is None. Value should be a string with both values separated by commas.",
    ),
    x_label: str = Option(
        None,
        "--x-label",
        "-xl",
        help="The label of the x axis. The default is time (t).",
    ),
    transient: int = Option(
        1000,
        "--transient",
        "-tr",
        help="The number of transient points discarded in the training of the model.",
    ),
    train_length: int = Option(
        10000,
        "--train-length",
        "-tl",
        help="The number of points used for the training of the model.",
    ),
):
    """Plot different data."""
    from functions import plot
    plot(
        plot_type=plot_type,
        predictions=predictions,
        data_file=data_file,
        lyapunov_exponent=lyapunov_exponent,
        delta_time=delta_time,
        plot_points=plot_points,
        title=title,
        save_path=save_path,
        show=show,
        y_labels=y_labels,
        y_values=y_values,
        x_label=x_label,
        transient=transient,
        train_length=train_length,
    )


if __name__ == "__main__":
    app()
