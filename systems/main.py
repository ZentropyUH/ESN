#!/usr/bin/python3
"""Cli interface to generate data of different physical models."""
# pylint: disable=invalid-name
import click
from functions import _lorenz, _mackey, _kuramoto, _rossler

# region lorenz params
@click.group()
@click.version_option(version="1.0.0")
def cli():
    """Train and predict with general ESN-like models from provided dynamical systems timeseries."""


@cli.command()
@click.option(
    "--initial-condition",
    "-ic",
    default=None,
    nargs=3,
    type=click.FLOAT,
    help="The initial conditions of the system. Default is randomly generated as [-10, 10]^3",
)
@click.option(
    "--delta-t",
    "-dt",
    default=0.02,
    type=click.FLOAT,
    help="The timestep length for the integration. Default is 0.02",
)
@click.option(
    "--steps",
    "-st",
    default=150000,
    type=click.INT,
    help="The timesteps to be taken for the integration. Default is 150000",
)
@click.option(
    "--final-time",
    "-ft",
    default=None,
    type=click.FLOAT,
    help="The final physical time for the integration. Default is None, infered from the steps",
)
@click.option(
    "--transient",
    "-tr",
    type=click.INT,
    default=0,
    help="Transient points to discard.",
)
@click.option(
    "--save-data",
    "-sv",
    default=False,
    is_flag=True,
    help="Save the data file or not. Default is False",
)
@click.option(
    "--plot",
    "-pl",
    default=False,
    is_flag=True,
    help="Choose whether to plot or not the function. Default is False",
)
@click.option(
    "--show",
    "-sh",
    default=False,
    is_flag=True,
    help="Choose whether to show the plot or not. Only makes sense when --plot. Default is False",
)
@click.option(
    "--seed",
    "-sd",
    default=None,
    type=click.INT,
    help="Choose the seed for random initial conditions. Default is None, random seed",
)
@click.option(
    "--plot-points",
    "-pp",
    default=2500,
    type=click.INT,
    help="The number of points to be plotted. Default is 2500",
)
@click.option(
    "--runs",
    "-r",
    type=click.INT,
    default=1,
    help="Number of runs. Only makes sense if random initial conditions are set. Default is 1",
)
# endregion
def lorenz(
    initial_condition,
    delta_t,
    steps,
    final_time,
    transient,
    save_data,
    plot,
    show,
    plot_points,
    seed,
    runs,
):
    """Integrate a Lorenz model with the given parameters. Data can be saved and plotted."""
    _lorenz(**locals())

# region mackey params
@cli.command()
@click.option(
    "--tau",
    "-t",
    type=click.FLOAT,
    default=16.8,
    help="Choose the parameter tau which controls the memory of the system. Default is 16.8",
)
@click.option(
    "--steps",
    "-st",
    type=click.INT,
    default=250000,
    help="The amount of timesteps to be taken in the integration. Default is 250000",
)
@click.option(
    "--delta-t",
    "-dt",
    type=click.FLOAT,
    default=0.05,
    help="Choose the timestep length for the integration. Default is 0.05",
)
@click.option(
    "--initial-condition",
    "-ic",
    default=None,
    nargs=1,
    type=click.FLOAT,
    help="Choose the initial condition for the system. Default is randomly generated in [0,1].",
)
@click.option(
    "--final-time",
    "-ft",
    default=None,
    type=click.FLOAT,
    help="Choose the final physical time for the integration.",
)
@click.option(
    "--save-data",
    "-sv",
    default=False,
    is_flag=True,
    help="Whether to save the generated data in a .csv file.",
)
@click.option(
    "--plot",
    "-pl",
    default=False,
    is_flag=True,
    help="Choose whether to plot or not the function.",
)
@click.option(
    "--show",
    "-sh",
    default=False,
    is_flag=True,
    help="Choose whether to show the plot or not. Only makes sense when --plot is given.",
)
@click.option(
    "--seed",
    "-sd",
    default=None,
    type=click.INT,
    help="Choose the seed for random initial conditions.",
)
@click.option(
    "--transient",
    "-tr",
    type=click.INT,
    default=0,
    help="Transient points to discard.",
)
@click.option(
    "--plot-points",
    "-pp",
    default=50000,
    type=click.INT,
    help="The amount of points to be plotted. Default is 2500",
)
@click.option(
    "--runs",
    "-r",
    type=click.INT,
    default=1,
    help="Number of runs. Only makes sense if random initial conditions are set.",
)
# endregion
def mackey(
    tau,
    steps,
    delta_t,
    initial_condition,
    final_time,
    save_data,
    transient,
    plot,
    show,
    plot_points,
    runs,
    seed,
):
    """Integrate a Mackey-Glass model with the given parameters. Data can be saved and plotted."""
    _mackey(**locals())

# region kuramoto params
@cli.command()
@click.option(
    "--spatial-period",
    "-l",
    default=22,
    type=click.INT,
    help="The length of the spatial interval for the KS model. Default is 22",
)
@click.option(
    "--discretization",
    "-n",
    default=64,
    type=click.INT,
    help="The number of discretization points along the spatial dimension. Default is 64",
)
@click.option(
    "--steps",
    "-st",
    type=click.INT,
    default=250000,
    help="The amount of timesteps in the integration. Default is 250000",
)
@click.option(
    "--delta-t",
    "-dt",
    type=click.FLOAT,
    default=0.05,
    help="Choose the timestep length for the integration.",
)
@click.option(
    "--initial-condition",
    "-ic",
    type=click.Choice(["random", "zeroes"]),
    default="random",
    help="Choose the initial condition. Default is randomly from interval [0,1]^N.",
)
@click.option(
    "--final-time",
    "-ft",
    default=None,
    type=click.FLOAT,
    help="Choose the final physical time for the integration.",
)
@click.option(
    "--save-data",
    "-sv",
    default=False,
    is_flag=True,
    help="Whether to save the generated data in a .csv file.",
)
@click.option(
    "--plot",
    "-pl",
    default=False,
    is_flag=True,
    help="Choose whether to plot or not the function.",
)
@click.option(
    "--show",
    "-sh",
    default=False,
    is_flag=True,
    help="Choose whether to show the plot or not. Only makes sense when --plot is given.",
)
@click.option(
    "--seed",
    "-sd",
    default=None,
    type=click.INT,
    help="Choose the seed for random initial conditions.",
)
@click.option(
    "--transient",
    "-tr",
    type=click.INT,
    default=0,
    help="Transient points to discard.",
)
@click.option(
    "--plot-points",
    "-pp",
    default=20000,
    type=click.INT,
    help="The number of points to be plotted. Default is 20000",
)
@click.option(
    "--runs",
    "-r",
    type=click.INT,
    default=1,
    help="Number of runs. Only makes sense if random initial conditions are set.",
)
# endregion
def kuramoto(
    spatial_period,
    discretization,
    delta_t,
    initial_condition,
    steps,
    final_time,
    save_data,
    transient,
    plot,
    plot_points,
    show,
    seed,
    runs,
):
    """Integrate a KS model with the given parameters. Data can be saved and plotted."""
    _kuramoto(**locals())

# region rossler params
@cli.command()
@click.option(
    "--steps",
    "-st",
    type=click.INT,
    default=250000,
    help="The amount of timesteps in the integration. Default is 250000",
)
@click.option(
    "--delta-t",
    "-dt",
    type=click.FLOAT,
    default=0.02,
    help="Choose the timestep length for the integration.",
)
@click.option(
    "--initial-condition",
    "-ic",
    type=click.Choice(["random", "zeroes"]),
    default=None,
    help="Choose the initial condition. Default is randomly from [-1,1]^3.",
)
@click.option(
    "--final-time",
    "-ft",
    default=None,
    type=click.FLOAT,
    help="Choose the final physical time for the integration.",
)
@click.option(
    "--save-data",
    "-sv",
    default=False,
    is_flag=True,
    help="Whether to save the generated data in a .csv file.",
)
@click.option(
    "--plot",
    "-pl",
    default=False,
    is_flag=True,
    help="Choose whether to plot or not the function.",
)
@click.option(
    "--show",
    "-sh",
    default=False,
    is_flag=True,
    help="Choose whether to show the plot or not. Only makes sense when --plot is given.",
)
@click.option(
    "--seed",
    "-sd",
    default=None,
    type=click.INT,
    help="Choose the seed for random initial conditions.",
)
@click.option(
    "--plot-points",
    "-pp",
    default=20000,
    type=click.INT,
    help="The number of points to be plotted. Default is 20000",
)
@click.option(
    "--runs",
    "-r",
    type=click.INT,
    default=1,
    help="Number of runs. Only makes sense if random initial conditions are set.",
)
@click.option(
    "--A", type=click.FLOAT, default=0.1, help="Parameter A of the model"
)
@click.option(
    "--B", type=click.FLOAT, default=0.1, help="Parameter B of the model"
)
@click.option(
    "--C", type=click.FLOAT, default=4, help="Parameter C of the model"
)
@click.option(
    "--transient",
    "-tr",
    type=click.INT,
    default=0,
    help="Transient points to discard.",
)
# endregion
def rossler(
    initial_condition,
    a,
    b,
    c,
    transient,
    delta_t,
    steps,
    final_time,
    save_data,
    plot,
    show,
    plot_points,
    seed,
    runs,
):
    """Integrate a Lorenz model with the given parameters. Data can be saved and plotted."""
    _rossler(**locals())

if __name__ == "__main__":
    cli()
