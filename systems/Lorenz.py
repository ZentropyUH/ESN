import random
from tkinter import PROJECTING
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint



rho = 28.0
sigma = 10.0
beta = 8.0 / 3.0

def f(state, t):
    """Return the time-derivative of a Lorenz system."""
    x, y, z = state  # Unpack the state vector
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # Derivatives

def gen_lorenz():
    random_states = [ ]

    for i in range (3):
        random_states.append(random.randint(0, 10))

    print(random_states)

    state0 = random_states
    dt = 0.02
    steps = 30000
    t = np.arange(0.0, steps, dt)

    states = odeint(f, state0, t)

    print(states.shape)

    x = states[:, 0]
    y = states[:, 1]
    z = states[:, 2]

    # # Plot x,y,z on different subplots
    # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    # ax1.plot(t, x)
    # ax1.set_ylabel("x")
    # ax2.plot(t, y)
    # ax2.set_ylabel("y")
    # ax3.plot(t, z)
    # ax3.set_ylabel("z")
    # ax3.set_xlabel("t")
    # plt.show()


    # Export the data as a Dataframe
    df = pd.DataFrame({"x": x, "y": y, "z": z})
    df.to_csv(
        f"data/Lorenz/Lorenz_initcond{state0}_rho{rho}_sigma{sigma}_beta{beta}_dt{dt}_steps{steps}.csv",
        index=False,
    )

    # # return map for z_t and z_t+1
    # fig, ax = plt.subplots()
    # ax.scatter(z[:-1], z[1:], s=0.5)
    # ax.set_xlabel("z_t")
    # ax.set_ylabel("z_t+1")
    # plt.show()


    pnts = 2500

    #fig = plt.figure()
    #ax = fig.gca()
    #ax.plot(states[:pnts, 0], states[:pnts, 1], states[:pnts, 2])
    #plt.draw()
    #plt.show()

for i in range(31):
    gen_lorenz()