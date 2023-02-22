"""Generate logistic map data."""
import os
import random
import click


def log_map(steps=100, initial=None, r_initial=2, delta_r=0):
    """Generate non stationary logistic map varying the parameter r."""
    if initial is None:
        initial = random.random()
    ans = [initial]
    r_factors = [r_initial]

    for _ in range(steps):
        nextv = r_factors[-1] * ans[-1] * (1 - ans[-1])
        ans.append(nextv)
        r_factors.append(r_factors[-1] + delta_r)
    return r_factors, ans


@click.group()
def home():
    """Generate logistic map data."""


# Change the argparse to click implementation
@home.command()
@click.option(  # -s STEPS -i INITIAL_CONDITION -r RFACTOR
    "-s",
    "--steps",
    default=100,
    required=True,
    help="Amount of time-steps to calculate the logisitc map",
    type=int,
)
@click.option(
    "-i",
    "--initial",
    default=None,
    required=False,
    help="Initial condition of the logisitc map.",
    type=float,
)
@click.option(
    "-r",
    "--r_factor",
    default=2,
    help="Value of 'r', the factor for the logisitc map.",
    type=float,
)
@click.option(
    "-d",
    "--delta_r",
    help="Delta_r value for the non stationary model.",
    default=0,
    type=float,
    required=False,
)
@click.option(
    "-o", "--outdir", help="Output folder.", type=str, default="./data/"
)
def map_me(steps, initial, r_factor, delta_r, outdir):
    """Generate logistic map."""
    # Check if the outdir exists
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Generate the logistic map
    r_, ans = log_map(
        steps=steps, initial=initial, r_initial=r_factor, delta_r=delta_r
    )

    # Save the logistic map
    name = f"LogisticMap_r_{r_factor}_delta_{delta_r}_steps_{steps}_initial_{initial}.csv"
    with open(os.path.join(outdir, name), "w", encoding="utf-8") as file:
        file.write("\n".join(map(str, ans)))


if __name__ == "__main__":
    home()
