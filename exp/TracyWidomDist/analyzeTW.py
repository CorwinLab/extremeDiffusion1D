import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import glob

files = glob.glob("/home/jhass2/Data/1.0/TracyWidom/T*.txt")
with open(files[0]) as g:
    vs = g.readline().split(",")[1:]
    vs = [float(i) for i in vs]

squared_sum = None
reg_sum = None

for f in files:
    data = np.loadtxt(f, delimiter=",", skiprows=1)
    time = data[:, 0]
    data = data[:, 1:]
    data = np.log(data / 1e300)  # subtract of N=1e300

    if squared_sum is None:
        squared_sum = data ** 2
    else:
        squared_sum += data ** 2

    if reg_sum is None:
        reg_sum = data
    else:
        reg_sum += data

mean = reg_sum / len(files)
var = squared_sum / len(files) - mean ** 2
np.savetxt("/home/jhass2/Data/1.0/TracyWidom/mean.txt", mean)
np.savetxt("/home/jhass2/Data/1.0/TracyWidom/var.txt", var)

for i in range(len(vs)):
    v = vs[i]
    v_var = var[:, i]
    I = 1 - np.sqrt(1 - v ** 2)
    sigma = (2 * I ** 2 / (1 - I)) ** (1 / 3)
    theory = time ** (2 / 3) * sigma ** 2
    fig, ax = plt.subplots()
    ax.set_xlabel("Time")
    ax.set_ylabel("Variance of Probability")
    ax.plot(time, v_var, label="Data")
    ax.plot(time, theory, label="Theory")
    ax.plot(time, time ** (1 / 2) * sigma ** 2, label="1/2 Power")
    ax.set_title(f"v={v} & {len(files)} Systems")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    fig.savefig(f"./figures/Variance{v}.png")
    plt.close(fig)

for i in range(len(vs)):
    v = vs[i]
    I = 1 - np.sqrt(1 - v ** 2)
    sigma = (2 * I ** 2 / (1 - I)) ** (1 / 3)
    theory = -I * time + time ** (1 / 3) * sigma * -1.77
    fig, ax = plt.subplots()
    ax.set_xlabel("Time")
    ax.set_ylabel("Mean Probability")
    ax.plot(time, abs(theory), label="Theory")
    ax.plot(time, abs(mean[:, i]), label="Data")
    ax.set_title(f"v={v} & {len(files)} Systems")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    fig.savefig(f"./figures/Mean{v}.png")
    plt.close(fig)
