from typer import Typer
from typer import Option

from cli.enums import *

app = Typer(
    help="",
    no_args_is_help=True,
)
@app.command(
    name="forecast",
    no_args_is_help=True,
)
def forecast_ESN(
    trained_model_path: str = Option(
        ...,
        "--trained-model-path",
        "-tm",
        help="The trained model to be used for forecasting",
    ),
    data_file: str = Option(
        ...,
        "--data-file",
        "-df",
        help="The data file to be used for training the model",
    ),
    output_dir: str = Option(
        ...,
        "--output-dir",
        "-o",
        help="The output directory where the forecasted data will be saved",
    ),
    forecast_method: EnumForecastMethod = Option(
        "classic",
        "--forecast-method",
        "-fm",
        help="The method to be used for forecasting. The default is Classic.",
        autocompletion=EnumForecastMethod.list,
    ),
    forecast_length: int = Option(
        1000,
        "--forecast-length",
        "-fl",
        help="The number of points to be forecasted. The default is 1000.",
    ),
    section_initialization_length: int = Option(
        50,
        "--section-initialization-length",
        "-sil",
        help="The number of points to be used for initializing the sections with true data. The default is 50.",
    ),
    number_of_sections: int = Option(
        10,
        "--number-of-sections",
        "-nos",
        help="The number of sections to be used for forecasting. The default is 10.",
    ),
    internal_states: bool = Option(
        False,
        "--internal-states/--no-show",
        "-is",
        help="Whether to save the inernal state values over the time or not. The default is False.",
    ),
    feedback_metrics: bool = Option(
        True,
        "--feedback",
        "-fb",
        help="Whether to include comparison metrics with the original data. Default is True.",
    ),
):
    """Make predictions with a given model on a data file."""
    from functions import forecast_from_saved_model
    forecast_from_saved_model(
        trained_model_path=trained_model_path,
        data_file=data_file,
        output_dir=output_dir,
        forecast_method=forecast_method,
        forecast_length=forecast_length,
        section_initialization_length=section_initialization_length,
        number_of_sections=number_of_sections,
        internal_states=internal_states,
        feedback_metrics=feedback_metrics
    )


if __name__ == "__main__":
    app()
