import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from matplotlib.animation import FuncAnimation
from matplotlib.pyplot import cm
from tqdm import trange

try:  # pyfftw is *much* faster
    from pyfftw.interfaces import cache, numpy_fft

    # print('# using pyfftw...')
    cache.enable()
    rfft = numpy_fft.rfft
    irfft = numpy_fft.irfft
except ImportError:  # fall back on numpy fft.
    print(
        "# WARNING: using numpy fft (install pyfftw for better performance)..."
    )

    def rfft(*args, **kwargs):
        kwargs.pop("threads", None)
        return np.fft.rfft(*args, **kwargs)

    def irfft(*args, **kwargs):
        kwargs.pop("threads", None)
        return np.fft.irfft(*args, **kwargs)


class KS(object):
    #
    # Solution of 1-d Kuramoto-Sivashinsky equation, the simplest
    # PDE that exhibits spatio-temporal chaos
    # (https://www.encyclopediaofmath.org/index.php/Kuramoto-Sivashinsky_equation).
    #
    # u_t + u*u_x + u_xx + diffusion*u_xxxx = 0, periodic BCs on [0,L].
    # time step dt with N fourier collocation points.
    # energy enters the system at long wavelengths via u_xx,
    # (an unstable diffusion term),
    # cascades to short wavelengths due to the nonlinearity u*u_x, and
    # dissipates via diffusion*u_xxxx.
    #
    def __init__(
        self,
        L=16,
        N=128,
        dt=0.5,
        diffusion=1.0,
        initial_conditions=None,
        rs=None,
    ):
        self.L = L
        self.n = N
        self.dt = dt

        self.diffusion = diffusion

        kk = N * np.fft.fftfreq(N)[0 : (N // 2) + 1]  # wave numbers
        self.wavenums = kk
        k = (
            kk.astype(float) * (2 * np.pi) / L
        )  # Made a correction here to make periodicity on [0,L] instead of [0,2*pi*L]

        self.ik = 1j * k  # spectral derivative operator

        self.lin = (
            k**2 - diffusion * k**4
        )  # Fourier multipliers for linear term

        # random noise initial condition.
        if rs is None:
            rs = np.random.RandomState()

        if initial_conditions is None:
            x = 0.01 * rs.standard_normal(size=(1, N))
        else:
            x = initial_conditions
            assert x.shape == (
                1,
                N,
            ), "Initial conditions must be of shape (1, N)"

        # remove zonal mean from initial condition.
        self.x = x - x.mean()

        # spectral space variable
        self.xspec = rfft(self.x, axis=-1)

    def nlterm(self, xspec):
        # compute tendency from nonlinear term.
        x = irfft(xspec, axis=-1)
        return -0.5 * self.ik * rfft(x**2, axis=-1)

    def advance(self):
        # semi-implicit third-order runge kutta update.
        # ref: http://journals.ametsoc.org/doi/pdf/10.1175/MWR3214.1
        self.xspec = rfft(self.x, axis=-1)
        xspec_save = self.xspec.copy()

        for n in range(3):
            dt = self.dt / (3 - n)

            # explicit RK3 step for nonlinear term
            self.xspec = xspec_save + dt * self.nlterm(self.xspec)

            # implicit trapezoidal adjustment for linear term
            self.xspec = (self.xspec + 0.5 * self.lin * dt * xspec_save) / (
                1.0 - 0.5 * self.lin * dt
            )
        self.x = irfft(self.xspec, axis=-1)


def generate_data(
    L=22,
    N=64,
    dt=0.25,
    cond0=None,
    steps=2000,
    t_end=None,
    diffusion=1,
    plot=False,
    plotpnts=2000,
    show=False,
    save=False,
    seed=None,
    name=None,
):
    # Overwiting the steps if t_end is given
    if t_end is not None:
        steps = int(t_end / dt)
        print("Total steps: ", steps)

    if seed is None:
        seed = np.random.randint(0, 1000000)

    rnd = np.random.default_rng(seed)

    y_discretization = np.linspace(0, L, N)

    match cond0:
        case "random":
            # Randomly choose the initial condition (These are close to 0, TODO Study this)
            state = 0.01 * rnd.random((1, N))
        case "zeroes":
            state = np.zeros((1, N))

    system = KS(
        L, N, dt, diffusion=diffusion, initial_conditions=state, rs=rnd
    )

    # system.xspec[0] = np.fft.rfft(state)

    timeseries = []
    timesteps = []

    for _ in trange(steps):
        system.advance()
        state = system.x.squeeze()
        timeseries.append(state)
        timesteps.append(_ * dt)

    timeseries = np.array(timeseries)
    timesteps = np.array(timesteps)

    name = f"KS_L{L}_N{N}_diffusion-k{diffusion}_dt{dt}_steps{steps}_seed{seed}.csv"

    if save:
        if not os.path.exists(f"data/KS/{L}"):
            os.makedirs(f"data/KS/{L}")

        df = pd.DataFrame(timeseries)

        df.to_csv(
            os.path.join(f"./data/KS/{L}", name), index=False, header=False
        )

    if plot:
        if not os.path.exists(f"data/KS/{L}"):
            os.makedirs(f"data/KS/{L}")

        fig = plt.figure()
        ax = fig.add_subplot()

        ax.contourf(
            timesteps[:plotpnts],
            y_discretization,
            timeseries.T[:, :plotpnts],
            levels=30,
        )

        with open(f"data/KS/{L}/" + name + ".pickle", "wb") as saved_plot:
            pickle.dump(fig, saved_plot)

        if show:
            plt.show()


def main():
    generate_data(plot=True, show=True)


if __name__ == "__main__":
    main()
