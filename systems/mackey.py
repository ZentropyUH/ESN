import numpy as np
import matplotlib.pyplot as plt
import math
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
        """The Mackey-Glass equation derivative.

        Args:
            y_t (float): y(t) The function value at time t.

            y_t_minus_tau (float): y(t - tau) The function value at time t - tau.

        Returns:
            float: y'(t) The function's derivative value at time t.

        """
        return -self.gamma * y_t + self.alpha * y_t_minus_tau / (
            1 + y_t_minus_tau**self.beta
        )

    def rk4(self, y_t, y_t_minus_tau, delta_t):
        """Runge-Kutta 4th order method.

        Args:
            y_t (float): the function value at time t.

            y_t_minus_tau (float): the function value at time t - tau.

            delta_t (float): the time step.

        Returns:
            float: y(t + delta_t) The function value at time t + delta_t.
        """
        k1 = delta_t * self.f(y_t, y_t_minus_tau)
        k2 = delta_t * self.f(y_t + 0.5 * k1, y_t_minus_tau)  # + delta_t*0.5
        k3 = delta_t * self.f(y_t + 0.5 * k2, y_t_minus_tau)  # + delta_t*0.5
        k4 = delta_t * self.f(y_t + k3, y_t_minus_tau)  # + delta_t
        return y_t + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6

    def gen(self, y0=0.5, delta_t=1, n=160000):
        """Generate the time-series.

        Args:
            y0 (float): the initial function value.

            delta_t (float): the time step.

            n (int): the number of time steps.

        Returns:
            tuple: (Y, T, X) The time-series, time steps, and delayed time-series.
        """
        time = 0
        index = 1
        history_length = math.floor(self.tau / delta_t)
        y_history = np.full(history_length, 0.5)
        y_t = y0
        y_t_ = 0
        Y = np.zeros(n + 1)
        X = np.zeros(n + 1)
        T = np.zeros(n + 1)

        for i in trange(n + 1):
            Y[i] = y_t
            X[i] = y_t_
            time = time + delta_t
            T[i] = time
            if self.tau == 0:
                y_t_minus_tau = y0
            else:
                y_t_minus_tau = y_history[index]

            y_t_plus_delta = self.rk4(y_t, y_t_minus_tau, delta_t)
            # print(y_t, y_t_minus_tau, y_t_plus_delta, time)
            if self.tau != 0:
                y_history[index] = y_t_plus_delta
                index = (index + 1) % history_length
            y_t_ = y_t
            y_t = y_t_plus_delta

        return Y, T, X

    def plot(self, discard=250 * 10):
        Y, T, X = self.gen()
        Y = Y[discard:]
        T = T[discard:]
        X = X[discard:]
        # plt.plot(Y[:-tau], Y[tau:])
        plt.plot(Y[2000 - self.tau : 2500 - self.tau], Y[2000:2500])
        # plt.plot(Y[2000:2500], Y[2000-self.tau:2500-self.tau]) #reverse x,y
        # plt.plot(Y[2000:2500], X[2000:2500])
        plt.title(
            "Mackey-Glass delay differential equation, tau = {}".format(
                self.tau
            )
        )
        plt.xlabel(r"$x(t - \tau)$")
        plt.ylabel(r"$x(t)$")
        plt.show()


mc = MackeyGlass(tau=22)
n = 250000
y, t, x = mc.gen(delta_t=1, n=n)
mc.plot()
# print((np.max(y) - np.min(y))/100, np.std(y), 0.03*np.std(y))
# print(np.max(y))

# # save to pandas dataframe
# df = pd.DataFrame({"x": x, "y": y})
# df.to_csv(
#     f"data/mackey_alpha{mc.alpha}_beta{mc.beta}_gamma{mc.gamma}_tau{mc.tau}_n{n}.csv",
#     index=False,
# )
