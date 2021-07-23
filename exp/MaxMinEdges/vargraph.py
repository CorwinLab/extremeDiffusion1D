import numpy as np
import glob
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

files = glob.glob("/home/jhass2/Data/*/1.00e_50/max_variance.txt")
n25files = glob.glob("/home/jhass2/Data/*/1.00e_25/max_variance.txt")


def getnonzeros(files):
    betas = []
    nonzeros = []

    for file in files:
        splits = file.split("/")
        beta = splits[4]
        if beta == "0.01":
            continue
        elif beta == "Einstein":
            continue
        betas.append(float(beta))
        var = np.loadtxt(file)
        first_nonzero = np.nonzero(var > 0.01)[0][0]
        nonzeros.append(first_nonzero)

    return betas, nonzeros


betas, nonzeros = getnonzeros(files)
betas = np.array(betas)
beta25, nonzeros25 = getnonzeros(n25files)
beta25 = np.array(beta25)
fig, ax = plt.subplots()
ax.set_xlabel("1/Beta")
ax.set_ylabel("First Time of Nonzero Variance/Log(N)")
ax.scatter(
    1 / betas, nonzeros / np.log2(float(int(1e50))), c="k", label="N=1e50"
)  # I think I originally did int(1e50) and so it approximated 1e50 with some extra change
ax.scatter(1 / beta25, nonzeros25 / np.log2(1e25), c="r", label="N=1e25")
ax.grid(True)
ax.set_yscale("log")
ax.legend()
fig.savefig("./figures/NonzeroTimes.png")
