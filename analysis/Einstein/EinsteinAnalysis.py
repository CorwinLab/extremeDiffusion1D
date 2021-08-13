import numpy as np
import glob
import os
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

topdir = "/home/jhass2/Data/Einstein/1.00e_50/*.txt"
files = glob.glob(topdir)


def mean_max_from_files(files):
    if not files:
        return
    top_dir = os.path.dirname(files[0])
    mean_save = os.path.join(top_dir, "max_mean.txt")
    var_save = os.path.join(top_dir, "max_variance.txt")

    all_data = None

    for file in files:
        data = np.loadtxt(file)
        center = np.arange(1, len(data[:, 0]) + 1) * 0.5
        min_edge = data[:, 0]
        max_edge = data[:, 1]
        minDist = abs(min_edge - center)
        maxDist = abs(max_edge - center)
        distance = np.max(np.vstack((minDist, maxDist)), 0)
        if all_data is None:
            all_data = distance
        else:
            all_data = np.vstack((all_data, distance))
        print(file)

    mean = np.mean(all_data, 0)
    var = np.var(all_data, 0)
    np.savetxt(mean_save, mean)
    np.savetxt(var_save, var)


if __name__ == "__main__":
    var_file = "/home/jhass2/Data/Einstein/1.00e_50/max_variance.txt"
    var = np.loadtxt(var_file)
    steps = np.arange(1, len(var) + 1)
    logN = np.log2(1e50)
    fig, ax = plt.subplots()
    ax.plot(steps, var, label="250 Systems")
    ax.plot([logN, logN], [min(var), max(var)], c="r", label="Log2(1e50)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Variiance")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("N=1e50, Einstein")
    ax.grid(True)
    ax.legend()
    fig.savefig("EinsteinVar.png")
