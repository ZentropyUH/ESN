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


# A = 0.1
# B = 0.1
# C = 4


def rossler_dydt(_t, y, A=10, B=28, C=2.667):
    xp = -y[1] - y[2]
    yp = y[0] + A * y[1]
    zp = B + y[2] * (y[0] - C)

    return np.asarray([xp, yp, zp])




def integrate(
    cond0=None,
    A=0.1,
    B=0.1,
    C=4,
    dt=0.02,
    steps=30000,
    t_end=None,
    save=False,
    plot=False,
    show=False,
    plotpnts=2500,
    seed=None,
    transient=0,
):
    
    rossler_f = partial(rossler_dydt, A=A, B=B, C=C)
    
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
        rossler_f, [0, t_end], state0, t_eval=timesteps, rtol=1e-12
    )

    states = states["y"]

    # eliminate transient
    states = states[:, transient:]

    print("states.shape")
    print(states.shape)

    name = f"Rossler_dt{dt}_steps{steps}_t-end{t_end}_seed{seed}"

    if save:
        x = states[0, :]
        y = states[1, :]
        z = states[2, :]

        if not os.path.exists(f"data/Rossler/{C}"):
            os.makedirs(f"data/Rossler/{C}")
        df = pd.DataFrame({"x": x, "y": y, "z": z})
        df.to_csv("data/Rossler/" + name + ".csv", index=False, header=False)

    if plot:
        if not os.path.exists("data/Rossler"):
            os.makedirs("data/Rossler")
            
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        ax.plot3D(
            states[0, :plotpnts],
            states[1, :plotpnts],
            states[2, :plotpnts],
        )

        with open("data/Rossler/" + name + ".pickle", "wb") as saved_plot:
            pickle.dump(fig, saved_plot)

        if show:
            plt.show()

    # return x, y, z


if __name__ == "__main__":
    integrate(
        steps=50000,
        plot=True,
        show=True,
        plotpnts=10000,
        transient=10000,
    )
