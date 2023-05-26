import typer
from enum import Enum

from src.grid.grid_one import *
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

@app.command()
def dnfj(
    path: str = typer.Option(..., '--path', '-p'),
    output: str = typer.Option(..., '--output', '-o'),
):
    detect_not_fished_jobs(
        path,
        output,
    )


@app.command()
def mse(
    path: str = typer.Option(..., '--path', '-p'),
    data_path: str = typer.Option(..., '--data', '-d'),
    output: str = typer.Option(..., '--output', '-o'),
    tl: int = typer.Option(..., '--training-lenght', '-tl'),
    trancient: int = typer.Option(..., '--trancient', '-t'),
):
    calculate_mse(
        path,
        data_path,
        output,
        tl,
        trancient,
    )


@app.command()
def plot_data_forecast(
    data_path: str = typer.Option(..., '--data', '-d'),
    forecast_path: str = typer.Option(..., '--forecast', '-f'),
    output: str = typer.Option(..., '--output', '-o'),
    trancient: int = typer.Option(..., '--trancient', '-t'),
):
    forecast_data = [join(forecast_path, x) for x in listdir(forecast_path)]
    data: list[str] = [join(data_path, p) for p in listdir(data_path)]

    for i, values in enumerate([(read_csv(d), read_csv(f)) for d, f in  zip(data, forecast_data)]):
        dd, ff = values
        dd = dd[:trancient]
        plots_data_forecast(dd[:1000], ff, join(output, str(i)))

if __name__ == "__main__":
    app()    a = [1,2,3,4,5,6,7,8,9]
