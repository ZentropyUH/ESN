import typer
from enum import Enum

from src.grid.grid_one import grid_one, best_combinations, change_folders
from src.functions import training, forecasting


app = typer.Typer()


@app.command()
def grid(
    index: int = typer.Option(..., '--index', '-i'),
    data_path: str = typer.Option(..., '--data', '-d'),
    output_path: str = typer.Option(..., '--output', '-o'),
    units: int = typer.Option(9000, '--units', '-u'),
    training_lenght: int = typer.Option(20000, '--training-lenght', '-tl'),
):
    grid_one(
        index,
        data_path,
        output_path,
        units,
        training_lenght,
    )


@app.command()
def best_params(
    path: str = typer.Option(..., '--path', '-p'),
    output: str = typer.Option(..., '--output', '-o'),
    max: int = typer.Option(..., '--max', '-m'),
    threshold: float = typer.Option(..., '--threshold', '-t'),
):
    best_combinations(
        path,
        output,
        max,
        threshold,
    )


@app.command()
def cf(
    path: str = typer.Option(..., '--path', '-p'),
):
    change_folders(
        path,
    )


if __name__ == "__main__":
    app()