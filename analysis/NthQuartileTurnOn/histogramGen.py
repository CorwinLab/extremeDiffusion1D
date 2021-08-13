import numpy as np
import npquad
import sys

sys.path.append("../../src")
sys.path.append("../../cDiffusion")
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import os
import glob
from pydiffusion import loadArrayQuad

files = glob.glob("/home/jhass2/Data/1.0/QuartileLarge/*.txt")

with open(files[0], "r") as f:
    line = f.readline()
    Ns = line.split(",")[2:]
    Ns = [np.quad(N) for N in Ns]

shape = np.loadtxt(files[0], delimiter=",", skiprows=1).shape

final_time_data = np.empty((len(files), shape[1]))

for row, f in enumerate(files):
    data = loadArrayQuad(f, shape, delimiter=",", skiprows=1)
    final_time = data[-1, :]
    final_time_data[row] = final_time.astype(np.float64)
    print(f)

times = final_time_data[:, 0]
final_time_data = final_time_data[:, 2:]  # Just exclude the maxEdge column

for col in range(final_time_data.shape[1]):
    N = Ns[col]
    fig, ax = plt.subplots()
    data = final_time_data[:, col]
    ax.hist(data)
    fig.savefig(f"./figures/hist{col}.png")
