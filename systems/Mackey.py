"""Module for integrating the Mackey-Glass equations."""
import math
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import trange


class MackeyGlass:
    """
    Generate time-series using the Mackey-Glass equation.

    Equation is numerically integrated by using a fourth-order Runge-Kutta method
    """

    def __init__(self, alpha=0.2, beta=10, gamma=0.1, tau=17):
        """Initialize the parameters."""
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau

    def f(self, y_t, y_t_minus_tau):
        """Calculate the Mackey-Glass equation derivative.

        Args:
            y_t (float): y(t) The function value at time t.

            y_t_minus_tau (float): y(t - tau) The function value at time t - tau.

        Returns:
            float: y'(t) The function's derivative value at time t.

        """
        return -self.gamma * y_t + self.alpha * y_t_minus_tau / (
            1 + y_t_minus_tau**self.beta
        )

    def rk4(self, y_t, y_t_minus_tau, dt):
        """Runge-Kutta 4th order method.

        Args:
            y_t (float): the function value at time t.

            y_t_minus_tau (float): the function value at time t - tau.

            dt (float): the time step.

        Returns:
            float: y(t + dt) The function value at time t + dt.
        """
        k1 = dt * self.f(y_t, y_t_minus_tau)
        k2 = dt * self.f(y_t + 0.5 * k1, y_t_minus_tau)  # + dt*0.5
        k3 = dt * self.f(y_t + 0.5 * k2, y_t_minus_tau)  # + dt*0.5
        k4 = dt * self.f(y_t + k3, y_t_minus_tau)  # + dt
        return y_t + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6

    def integrate(
        self,
        dt=1,
        y0=None,
        steps=160000,
        t_end=None,
        plot=False,
        show=False,
        save=False,
        plotpnts=2500,
        seed=None,
    ):
        """Generate the time-series.

        Args:
            y0 (float): the initial function value.

            dt (float): the time step.

            n (int): the number of time steps.

        Returns:
            tuple: (Y, T, X) The time-series, time steps, and delayed time-series.
        """
        if seed is None:
            seed = np.random.randint(1000000)

        rnd = np.random.default_rng(seed=seed)

        if t_end != None:
            steps = int(t_end / dt)

        t_end = steps * dt

        if y0 is None:
            y0 = rnd.random()

        index = 1
        history_length = math.floor(self.tau / dt)
        y_history = np.full(history_length, y0)
        y_t = y0
        # y_t_ = 0
        Y = np.zeros(steps)
        # X = np.zeros(steps)

        for i in trange(steps):
            Y[i] = y_t
            # X[i] = y_t_
            if self.tau == 0:
                y_t_minus_tau = y0
            else:
                y_t_minus_tau = y_history[index]

            y_t_plus_delta = self.rk4(y_t, y_t_minus_tau, dt)
            # print(y_t, y_t_minus_tau, y_t_plus_delta, time)
            if self.tau != 0:
                y_history[index] = y_t_plus_delta
                index = (index + 1) % history_length
            # y_t_ = y_t
            y_t = y_t_plus_delta

        name = f"MG_tau{self.tau}_dt{dt}_n{steps}_t-end{t_end}_seed{seed}"

        if save:
            if not os.path.exists(f"data/MG/{self.tau}"):
                os.makedirs(f"data/MG/{self.tau}")
            df = pd.DataFrame({"y": Y})
            df.to_csv(
                f"data/MG/{self.tau}/" + name + ".csv",
                index=False,
                header=False,
            )

        if plot:
            if not os.path.exists(f"data/MG/{self.tau}"):
                os.makedirs(f"data/MG/{self.tau}")

            fig = plt.figure()
            ax = fig.add_subplot()

            x = np.arange(0, t_end, dt)

            ax.plot(x[:plotpnts], Y[:plotpnts])

            with open(
                f"data/MG/{self.tau}/" + name + ".pickle", "wb"
            ) as saved_plot:
                pickle.dump(fig, saved_plot)

            if show:
                plt.show()

        return Y


def main():
    tau = 17

    mc = MackeyGlass(tau=tau)
    y = mc.integrate(y0=0.5, dt=0.05, t_end=4000)

    # plt.plot(x)
    plt.plot(y)
    plt.show()


if __name__ == "__main__":
    main()
