from typer import Typer
from typer import Option
from typing import List

from cli.enums import *


app = Typer(
    help="",
    no_args_is_help=True,
)


@app.command(
    name="slurm_grid",
    no_args_is_help=True,
    help='Commad executed by the slurm grid search.'
)
def slurm_grid_command(
    data_path: str = Option(
        ...,
        "--data", "-d",
        help='Path to the system data.'
    ),
    output_path: str = Option(
        ...,
        "--output", "-o",
        help='Output path.'
    ),
    index: int = Option(
        ...,
        "--index", "-i",
        help='Index of the combination in the combinations.json file.'
    ),
    hyperparameters_path: str = Option(
        ...,
        "--hyperparameters-path", "-hp",
        help='Path to the .json file of the combinations.'
    ),
):
    from slurm_grid.grid import slurm_grid
    slurm_grid(
        data_path=data_path,
        output_path=output_path,
        index=index,
        hyperparameters_path=hyperparameters_path,
    )


@app.command(
    name="init_slurm_grid",
    no_args_is_help=True,
    help='Initialize all files and folders for grid search.'
)
def init_slurm_grid_command(
    path: str = Option(
        ...,
        '--path', '-p',
        help='Base path to save grid search folders and files.'
    ),
    job_name: str = Option(
        'job',
        '--job-name', '-j',
        help='Slurm job name.'
    ),
    data_path: str = Option(
        ...,
        '--data-path', '-dp',
        help='Path of the System data.'
    ),
    model: str = Option(
        'ESN',
        '-m', '--model'
    ),
    input_initializer: str = Option(
        'InputMatrix',
        '-ii', '--input-initializer'
    ),
    input_bias_initializer: str = Option(
        'RandomUniform',
        '-ib', '--input-bias'
    ),
    reservoir_activation: str = Option(
        'tanh', '-ra',
        '--reservoir-activation'
    ),
    reservoir_initializer: str = Option(
        'WattsStrogatzNX',
        '-ri', '--reservoir-initializer'
    ),
    units: List[int] = Option(
        [5000],
        '--units', '-u'
    ),
    train_length: List[int] = Option(
        [20000],
        '--train-length', '-tl'
    ),
    forecast_length: List[int] = Option(
        [1000],
        '--forecast-length', '-fl'
    ),
    transient: List[int] = Option(
        [1000],
        '--transient', '-t'
    ),
    steps: List[int] = Option(
        [1],
        '--steps', '-s'
    ),
    delta_time: List[float] = Option(
        [1],
        '--delta-time', '-dt'
    ),
    lyapunov_exponent: List[float] = Option(
        [1],
        '--lyapunov-exponent', '-ly'
    ),
    input_scaling: List[float] = Option(
        ...,
        '--input-scaling', '-is'
    ),
    leak_rate: List[float] = Option(
        ...,
        '--leak-rate', '-lr'
    ),
    spectral_radius: List[float] = Option(
        ...,
        '--spectral-radius', '-sr'
    ),
    rewiring: List[float] = Option(
        ...,
        '--rewiring', '-rw'
    ),
    reservoir_degree: List[int] = Option(
        ...,
        '--reservoir-degree', '-rd'
    ),
    reservoir_sigma: List[float] = Option(
        ...,
        '--reservoir-sigma', '-rs'
    ),
    regularization: List[float] = Option(
        ...,
        '--regularization', '-rg'
    ),
    reservoir_amount: List[int] = Option(
        ...,
        '--reservoir-amount', '-ra'
    ),
    overlap: List[int] = Option(
        ...,
        '--overlap', '-ol'
    ),
):
    from slurm_grid.tools import init_slurm_grid
    init_slurm_grid(
        path=path,
        job_name=job_name,
        data_path=data_path,
        model=model,
        input_initializer=input_initializer,
        input_bias_initializer=input_bias_initializer,
        reservoir_activation=reservoir_activation,
        reservoir_initializer=reservoir_initializer,
        units=units,
        train_length=train_length,
        forecast_length=forecast_length,
        transient=transient,
        steps=steps,
        dt=delta_time,
        lyapunov_exponent=lyapunov_exponent,
        input_scaling=input_scaling,
        leak_rate=leak_rate,
        spectral_radius=spectral_radius,
        rewiring=rewiring,
        reservoir_degree=reservoir_degree,
        reservoir_sigma=reservoir_sigma,
        regularization=regularization,
        reservoir_amount=reservoir_amount,
        overlap=overlap,
    )


# FIX
@app.command(
    name="grid_aux",
    no_args_is_help=True,
    help='Generate the next steps of the grid search from the results of the previous ones.',
)
def grid_aux_command(
    path: str = Option(
        ...,
        '--path', '-p',
        help='Base path to grid search folders'
    ),
    n_results: int = Option(
        ...,
        "--n-results", "-nr"
    ),
    threshold: float = Option(
        ...,
        "--threshold", "-t"
    ),
):
    # TODO: Adapt method to new changes
    raise NotImplementedError
    from slurm_grid.tools import grid_aux
    grid_aux(
        path=path,
        n_results=n_results,
        threshold=threshold,
    )


@app.command(
    name="best_results",
    no_args_is_help=True,
    help='Get the best results from the given path. Compare by the given `threshold`.',
)
def best_results_command(
    results_path: str = Option(..., "--results-path", "-rp"),
    output: str = Option(..., "--output", "-o"),
    n_results: int = Option(..., "--n-results", "-nr"),
    threshold: float = Option(..., "--threshold", "-t"),
):
    from slurm_grid.tools import best_results
    best_results(
        results_path=results_path,
        output=output,
        n_results=n_results,
        threshold=threshold,
    )


@app.command(
    name="results_data",
    no_args_is_help=True,
    help='Generate the a .json file with the hyperparameters of every training and the index where the rmse from the results are bigger than the threshold.',
)
def results_data_command(
    results_path: str = Option(..., "--results-path", "-rp", help='Path of the results from grid search to be analized.'),
    filepath: str = Option(..., "--filepath", "-fp", help='File path for the output. Must be a .json file.'),
    threshold: float = Option(..., "--threshold", "-t"),
):
    from slurm_grid.tools import results_data
    results_data(
        results_path=results_path,
        filepath=filepath,
        threshold=threshold,
    )
    

@app.command(
    name="search_unfinished_combinations",
    no_args_is_help=True,
    help='Search for the combinations that have not been satisfactorily completed and create a script to execute them.',
)
def search_unfinished_combinations_command(
    path:str =  Option(..., "--path", "-p", help='Specify the folder where the results of the combinations are stored'),
    depth = Option(0, "--depth", "-d", help='Grid depth, to specify the depth of the grid seach.')
):
    from slurm_grid.tools import search_unfinished_combinations
    search_unfinished_combinations(
        path=path,
        depth=depth,
    )


if __name__ == "__main__":
    app()
