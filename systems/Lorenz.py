import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
import pickle
from tqdm import tqdm
import os

from functools import partial


############################ MONKEY PATCH FOR PBAR ############################

from scipy.integrate._ivp.base import (
    OdeSolver,
)  # this is the class we will monkey patch
from tqdm import tqdm

### monkey patching the ode solvers with a progress bar

# save the old methods - we still need them
old_init = OdeSolver.__init__
old_step = OdeSolver.step


# define our own methods
def new_init(self, fun, t0, y0, t_bound, vectorized, support_complex=False):
    # define the progress bar
    self.pbar = tqdm(
        total=t_bound - t0, unit="ut", initial=t0, ascii=False, desc="IVP"
    )
    self.last_t = t0

    # call the old method - we still want to do the old things too!
    old_init(self, fun, t0, y0, t_bound, vectorized, support_complex)


def new_step(self):
    # call the old method
    old_step(self)

    # update the bar
    tst = self.t - self.last_t
    self.pbar.update(tst)
    self.last_t = self.t

    # close the bar if the end is reached
    if self.t >= self.t_bound:
        self.pbar.close()


# overwrite the old methods with our customized ones
OdeSolver.__init__ = new_init
OdeSolver.step = new_step

############################ MONKEY PATCH FOR PBAR ############################


SIGMA = 10.0
RHO = 28.0
BETA = 8.0 / 3.0


# def lorenz_f(state, _t):
#     """Return the time-derivative of a Lorenz system."""
#     x, y, z = state  # Unpack the state vector
#     return SIGMA * (y - x), x * (RHO - z) - y, x * y - BETA * z  # Derivatives


def lorenz_dydt(_t, y, sigma=10, rho=28, beta=2.667):
    xp = sigma * (y[1] - y[0])
    yp = y[0] * (rho - y[2]) - y[1]
    zp = y[0] * y[1] - beta * y[2]

    return np.asarray([xp, yp, zp])


lorenz_f = partial(lorenz_dydt, sigma=SIGMA, rho=RHO, beta=BETA)


def integrate(
    cond0=None,
    dt=0.02,
    steps=30000,
    t_end=None,
    save=False,
    plot=False,
    show=False,
    plotpnts=2500,
    seed=None,
):
    if seed is None:
        seed = np.random.randint(1000000)

    rnd = np.random.default_rng(seed=seed)

    if t_end != None:
        steps = int(t_end / dt)

    t_end = steps * dt

    if cond0 is None:
        cond0 = (rnd.random(3) - 0.5) * 2

    state0 = cond0

    timesteps = np.arange(0.0, t_end, dt)

    states = solve_ivp(
        lorenz_f, [0, t_end], state0, t_eval=timesteps, rtol=1e-12
    )

    states = states["y"]

    name = f"Lorenz_dt{dt}_steps{steps}_t-end{t_end}_seed{seed}"

    if save:
        x = states[0, :]
        y = states[1, :]
        z = states[2, :]

        if not os.path.exists("data/Lorenz"):
            os.makedirs("data/Lorenz")
        df = pd.DataFrame({"x": x, "y": y, "z": z})
        df.to_csv("data/Lorenz/" + name + ".csv", index=False, header=False)

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        ax.plot3D(
            states[0, :plotpnts], states[1, :plotpnts], states[2, :plotpnts]
        )

        with open("data/Lorenz/" + name + ".pickle", "wb") as saved_plot:
            pickle.dump(fig, saved_plot)

        if show:
            plt.show()

    # return x, y, z


if __name__ == "__main__":
    print("caca")
