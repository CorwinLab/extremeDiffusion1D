import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt

data = np.loadtxt("Data.txt", delimiter=",", skiprows=1)
data = data[1:, :]
times = data[:, 0]
maxEdge = data[:, 1]
Ns = [1e2, 1e5, 1e10, 1e15, 1e25]

for i, N in enumerate(Ns):
    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Time")
    ax.set_ylabel("Nth Quartile")
    Nstr = "{:.0e}".format(N)
    theory = np.piecewise(
        times,
        [times < np.log(N), times >= np.log(N)],
        [lambda x: x, lambda x: x * np.sqrt(1 - (1 - np.log(N) / x) ** 2)],
    )
    ax.plot(times / np.log(N), 2 * data[:, i + 2], label="N=" + Nstr + " Data")
    ax.plot(times / np.log(N), theory, label="Theoretical Curve")

    ax.legend()
    fig.savefig(f"thoery{i}.png")
