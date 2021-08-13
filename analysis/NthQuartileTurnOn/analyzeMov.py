import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

plt.rcParams.update({"font.size": 16})
import numpy as np
import glob
import sys

sys.path.append("../../src")
sys.path.append("../../cDiffusion")
from pydiffusion import loadArrayQuad, Diffusion
import os
from scipy.interpolate import interp1d

file_dir = "/home/jhass2/Data/1.0/QuartileLarge/"

files = glob.glob(file_dir + "Q*.txt")
print("Number of files found:", len(files))
with open(files[0]) as g:
    Ns = g.readline().split(",")[2:]
    Ns = [np.quad(N) for N in Ns]

data = np.loadtxt(files[0], delimiter=",", skiprows=1)
shape = data.shape

squared_sum = None
reg_sum = None

run_again = False

if not os.path.isfile(file_dir + "mean.txt") or run_again:
    count = 0
    for f in files:
        try:
            data = loadArrayQuad(f, shape, skiprows=1, delimiter=",")
        except Exception as e:
            print("File went wrong: ", f)
            print(e)
            continue

        time = data[:, 0]
        data = 2 * data[:, 2:]

        if squared_sum is None:
            squared_sum = data ** 2
        else:
            squared_sum += data ** 2

        if reg_sum is None:
            reg_sum = data
        else:
            reg_sum += data

        count += 1
        print(f)

    mean = reg_sum / count
    var = squared_sum / count - mean ** 2
    mean = mean.astype(np.float64)
    var = var.astype(np.float64)
    time = time.astype(np.float64)
    np.savetxt(file_dir + "mean.txt", mean)
    np.savetxt(file_dir + "var.txt", var)

else:
    mean = np.loadtxt(file_dir + "mean.txt")
    var = np.loadtxt(file_dir + "var.txt")
    data = loadArrayQuad(files[0], shape, skiprows=1, delimiter=",")
    time = data[:, 0]
    time = time.astype(np.float64)

exps = []
for N in Ns:
    N = str(N)
    exp = int(N.split("e")[-1])
    exps.append(round(exp, -1))

lin_fits = []
for col, N in enumerate(Ns):
    logN = np.log(N).astype(np.float64)
    t = time / logN
    v = var[:, col] / (logN ** (2 / 3))
    lin_fits.append(interp1d(t, v))


sample_times = np.linspace(1, max(t), 1000)
for i, t in enumerate(sample_times):
    logN = np.log(Ns).astype(np.float64)
    ts = t * logN
    theoretical = Diffusion.theoreticalNthQuartVar(Ns, ts)
    theoreticalv = theoretical / logN ** (2 / 3)
    vals = [fit(t) for fit in lin_fits]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(f"t={t}")
    ax.scatter(exps, vals)
    ax.set_xlabel("Exponent")
    ax.set_ylabel("Variance / Log(N)")
    ax.hlines(theoreticalv, min(exps), max(exps), label="Theory")
    ax.legend()
    fig.savefig(f"./mov/Frame{i}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(i / len(sample_times))

"""
for row in range(len(var)): 
    t = time[row]
    v = var[row, :]
    theoretical = Diffusion.theoreticalNthQuartVar(Ns, t)
    
    yscale = 1 / (np.log(Ns).astype(np.float64) ** (2/3))
    fig, ax = plt.subplots()
    ax.scatter(exps, v * yscale)
    ax.hlines(theoretical * yscale, min(exps), max(exps))
    ax.set_xlabel('Exponent')
    ax.set_ylabel('Variance')
    fig.savefig(f'./mov/Frame{row}.png')
    plt.close(fig)


"""
