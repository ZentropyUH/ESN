from typer import Typer
from typer import Option

from cli.enums import *


app = Typer(
    help="",
    no_args_is_help=True,
)


@app.command(
    name="KS-ly",
    no_args_is_help=True,
    help="Estimation of the i-th largest Lyapunov Time of the KS model, based on the paper: 'Lyapunov Exponents of the Kuramoto-Sivashinsky PDE. arxiv:1902.09651v1'",
)
def KS_ly(
    i_th: int = Option(
        1,
        "--i-th",
        "-i",
        help="The i-th largest Lyapunov Time of the KS model.",
    ),
    length_scale: int = Option(
        ...,
        "--length-scale",
        "-l",
        help="The length scale of the KS model.",
    ),
):
    from src.utils import lyap_ks
    value = lyap_ks(i_th, length_scale)
    print(f"Lyapunov Time of the KS model: {value}")


if __name__ == "__main__":
    app()
