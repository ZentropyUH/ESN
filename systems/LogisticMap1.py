"""Make bifurcation diagam of the logistic map."""
import numpy as np
import matplotlib.pyplot as plt

interval = (3.8, 3.9)  # start, end
accuracy = 0.00001
total_r = 1000

delta_r = 0.00001
delta_r = 0

reps = 600  # number of repetitions
numtoplot = 200
lims = np.zeros(reps)

fig, biax = plt.subplots()
fig.set_size_inches(16, 9)

lims[0] = np.random.rand()
for r in np.linspace(interval[0], interval[1], total_r):
    # for r in np.arange(interval[0], interval[1], accuracy):
    r_ = r
    for i in range(reps - 1):
        lims[i + 1] = r_ * lims[i] * (1 - lims[i])
        r_ += delta_r

    biax.scatter([r] * numtoplot, lims[reps - numtoplot :], s=0.02, c="blue")

biax.set(xlabel="r", ylabel="x", title="logistic map")
plt.show()
