import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 16})
import numpy as np
import glob
import sys
sys.path.append("../../src")
sys.path.append("../../cDiffusion")
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

run_again = False

if not os.path.isfile(file_dir + 'mean.txt') or run_again: 
    count = 0
    for f in files:
        try:
            data = loadArrayQuad(f, shape, skiprows=1, delimiter=",")
        except Exception as e:
            print('File went wrong: ', f)
            print(e)
            continue

        time = data[:, 0]
        data = data[:, 1:]
        data = np.log(data)

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

save_folder = 'figuresN8000'

for i in range(len(vs)):
    v = (vs[i])
    v_var = var[:, i]
    I = 1 - np.sqrt(1 - v ** 2)
    sigma = (2 * I ** 2 / (1 - I)) ** (1 / 3)
    theory = (time ** (2/3)) * sigma**2
    fig, ax = plt.subplots()
    ax.set_xlabel("Time")
    ax.set_ylabel("Var(ln(Pb(vt, t))")
    ax.plot(time, v_var, label="Data", c='r')
    ax.plot(time, theory, label=r"$ t^{2/3} * \sigma^{2}$", c='k')
    ax.set_title(f"v={v} & {len(files)} Systems")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    fig.savefig(f"./{save_folder}/Variance{v}.png", bbox_inches='tight')
    plt.close(fig)

for i in range(len(vs)):
    v = vs[i]
    I = 1 - np.sqrt(1 - v ** 2)
    sigma = (2 * I ** 2 / (1 - I)) ** (1 / 3)
    theory = -I * time + time ** (1 / 3) * sigma * -1.77
    fig, ax = plt.subplots()
    ax.set_xlabel("Time")
    ax.set_ylabel("|ln(Pb(vt, t))|")
    ax.plot(time, abs(mean[:, i]), label="Data", c='r')
    ax.plot(time, abs(theory), label=r"$|-I * t + t^{1/3} * \sigma * -1.77|$", c='k')
    ax.set_title(f"v={v} & {len(files)} Systems")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    fig.savefig(f"./{save_folder}/Mean{v}.png", bbox_inches='tight')
    plt.close(fig)
