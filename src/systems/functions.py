"""Functions to integrate chaotic models."""
# pylint: disable=invalid-name
import KS
import Lorenz as L_model
import Rossler as R_model
from mackey import MackeyGlass as MG_model
from tqdm import trange


def _lorenz(
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
    if not (save_data or plot):
        print(
            "Not saving the data nor plotting, not making a choice is itself choosing..."
        )
        exit(0)

    for _ in trange(runs, desc="Number of simulations"):
        L_model.integrate(
            cond0=initial_condition,
            dt=delta_t,
            steps=steps,
            t_end=final_time,
            transient=transient,
            save=save_data,
            plot=plot,
            show=show,
            plotpnts=plot_points,
            seed=seed,
        )

def _mackey(
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
    if not (save_data or plot):
        print(
            "Not saving the data nor plotting, not making a choice is itself choosing..."
        )
        exit(0)

    for _ in trange(runs, desc="Number of simulations"):
        model = MG_model(tau=tau)
        model.integrate(
            dt=delta_t,
            y0=initial_condition,
            t_end=final_time,
            steps=steps,
            save=save_data,
            transient=transient,
            plot=plot,
            show=show,
            plotpnts=plot_points,
            seed=seed,
        )

def _kuramoto(
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
    if not (save_data or plot):
        print(
            "Not saving the data nor plotting, not making a choice is itself choosing..."
        )
        exit(0)

    for _ in trange(runs, desc="Number of simulations"):
        KS.generate_data(
            L=spatial_period,
            N=discretization,
            dt=delta_t,
            cond0=initial_condition,
            steps=steps,
            t_end=final_time,
            save=save_data,
            transient=transient,
            plot=plot,
            plotpnts=plot_points,
            show=show,
            seed=seed,
        )

def _rossler(
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
    if not (save_data or plot):
        print(
            "Not saving the data nor plotting, not making a choice is itself choosing..."
        )
        exit(0)

    for _ in trange(runs, desc="Number of simulations"):
        R_model.integrate(
            cond0=initial_condition,
            A=a,
            B=b,
            C=c,
            transient=transient,
            dt=delta_t,
            steps=steps,
            t_end=final_time,
            save=save_data,
            plot=plot,
            show=show,
            plotpnts=plot_points,
            seed=seed,
        )
