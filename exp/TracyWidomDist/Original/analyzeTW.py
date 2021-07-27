import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 16})
import numpy as np
import glob
import sys
sys.path.append("../../../src")
sys.path.append("../../../cDiffusion")
from pydiffusion import loadArrayQuad
import os

file_dir = "/home/jhass2/Data/1.0/TracyWidomN8000/"

files = glob.glob(file_dir + "T*.txt")
print('Number of files found:', len(files))
with open(files[0]) as g:
    vs = g.readline().split(",")[1:]
    vs = [float(i) for i in vs]

data = np.loadtxt(files[0], delimiter=",", skiprows=1)
shape = data.shape

squared_sum = None
reg_sum = None

run_again = True

# rows are different systems and colums are different vs
total_data = np.empty((len(files), shape[1] - 1))

if run_again:
    count = 0
    for row, f in enumerate(files):
        try:
            data = loadArrayQuad(f, shape, skiprows=1, delimiter=",")
        except Exception as e:
            print('File went wrong: ', f)
            print(e)
            continue

        time = data[-1, 0].astype(np.float64)
        data = data[-1, 1:]
        data = np.log(data).astype(np.float64)
        total_data[row] = data

    np.savetxt(file_dir + 'FinalTime.txt')

else:
    np.loadtxt(file_dir + 'FinalTime.txt')


for i in range(len(vs)):
    v = vs[i]
    data = total_data[:, i]
    fig, ax = plt.subplots()
    ax.set_xlabel("ln(Pb(vt, t))")
    ax.set_ylabel("Counts")
    ax.set_title(f"v={v}")
    ax.hist(data - np.mean(data), density=True)
    fig.savefig(f"./histograms/hist{v}.png")
