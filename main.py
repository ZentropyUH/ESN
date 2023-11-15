#!/usr/bin/python3
from typer import Typer

from cli import app_aux
from cli import app_forecast
from cli import app_grid
from cli import app_plot
from cli import app_train


app = Typer(
    help="A tool for forecasting chaotic systems using Echo State Networks.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


app.add_typer(app_aux, name="aux")
app.add_typer(app_forecast, name="forecast")
app.add_typer(app_grid, name="grid")
app.add_typer(app_plot, name="plot")
app.add_typer(app_train, name="train")


if __name__ == "__main__":
    app()
