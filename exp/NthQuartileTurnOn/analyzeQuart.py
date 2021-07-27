import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 16})
import numpy as np
import glob
import sys
sys.path.append("../../src")
sys.path.append("../../cDiffusion")
from pydiffusion import loadArrayQuad, Diffusion
import os

file_dir = "/home/jhass2/Data/1.0/QuartileLarge/"

files = glob.glob(file_dir + "Q*.txt")
print('Number of files found:', len(files))
with open(files[0]) as g:
    Ns = g.readline().split(",")[2:]
    Ns = [np.quad(N) for N in Ns]

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

save_folder = './MeanVarFigures/'

for i in range(len(Ns)):
    N = Ns[i]
    Nstr = str(N)
    exp = Nstr.split('e')[1][1:]
    exp = round(int(exp), -1)
    exp_str = str(exp)

    data_mean = mean[:, i]
    data_var = var[:, i]
    theory_mean = Diffusion.theoreticalNthQuart(N, time)
    theory_var = Diffusion.theoreticalNthQuartVar(N, time)

    xscale = np.quad('1') / np.log(N)
    xscale = np.float64(xscale)
    fig, ax = plt.subplots(2, sharex=True, figsize=(12,12))
    ax[0].set_title(f"N=10^{exp}")
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_ylabel("Mean Nth Quartile")
    ax[0].plot(time * xscale, data_mean, c='r', label='Data')
    ax[0].plot(time * xscale, theory_mean, c='k', label='Theory')
    ax[0].legend()
    ax[1].plot(time * xscale, data_var, c='r', label='Data')
    ax[1].plot(time * xscale, theory_var, c='k', label='Theory')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlabel('Time/ln(N)')
    ax[1].set_ylabel('Var Nth Quartile')
    ax[1].legend()
    ax[1].set_xlim([0.9, max(time * xscale)])
    print('Maximum Time:', max(time))
    save_file = save_folder + f"MeanVar{exp}.png"
    print("Saving File at: ", save_file) 
    fig.savefig(save_file, bbox_inches="tight")
    shifted_time = abs(time - np.log(N).astype(np.float64))
    min_idx = np.argmin(shifted_time)
    print(f'Variance at t={time[min_idx]*xscale}: {data_var[min_idx]}')

