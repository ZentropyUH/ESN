import typer
from t_utils import *

from src.grid.grid_one import *
from src.functions import training, forecasting


app = typer.Typer()




@app.command()
def train(
    # General params
    model: EModel = typer.Option('ESN', '--model', '-m', help=''),
    units: str = typer.Option(..., '--units', '-u', help=''),
    input_initializer: EInputInitializer = typer.Option(
        'InputMatrix','--input-initializer', '-ii',
        help= '',
    ),
    # input_scaling: str = typer.O,
    # leak_rate,
    # reservoir_activation,
    # # Classic Cases
    # spectral_radius,
    # reservoir_initializer,
    # rewiring,
    # reservoir_degree,
    # reservoir_sigma,
    # # Parallel cases
    # reservoir_amount,
    # overlap,
    # # Readout params
    # readout_layer,
    # regularization,
    # # Training params
    # init_transient,
    # transient,
    # train_length,
    # data_file,
    # output_dir,
    # trained_name,
):
    print('Train')


@app.command()
def forecast():
    print('Forecast')


@app.command()
def plot():
    print('Plot')


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
        dd = dd[trancient:]
        plots_data_forecast(dd[:1000], ff, join(output, str(i)))


# for i in range(0, 24000, 1000):
    #     print(i)
    #     calculate_mse(
    #         '/home/dionisio35/Documents/a/0.2_6_1.0000000000000002e-08_0.99_0.4/forecast/',
    #         '/media/dionisio35/Windows/_folders/_new/Lorenz/',
    #         '/home/dionisio35/Documents/B/prediction/all/{}'.format(i),
    #         i,
    #     )



if __name__ == "__main__":
    app()