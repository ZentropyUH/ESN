from typer import Typer
from typer import Option

from cli.enums import *


app = Typer(
    help="",
    no_args_is_help=True,
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
