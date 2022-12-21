import numpy as np
from matplotlib import pyplot as plt 
import pandas as pd
import glob
import sys
sys.path.append("../../dataAnalysis")
from theory import quantileMean

run_again = False 
if run_again:
    beta_dir = "/home/jacob/Desktop/corwinLabMount/CleanData/CDFBetaSweep/100/Q*.txt"
    files = glob.glob(beta_dir)
    quantiles = None
    for f in files: 
        df = pd.read_csv(f)
        quantile = df['1.00000000000000000000000000000000004e+300'].values
        time = df['time'].values
        if quantiles is None: 
            quantiles = quantile 
        else: 
            quantiles = np.vstack((quantiles, quantile))
    np.savetxt("Quantiles.txt", quantiles)
    np.savetxt("Time.txt", time)
else:
    quantiles = np.loadtxt("Quantiles.txt")
    time = np.loadtxt("Time.txt")

run_again = False
if run_again:
    quantiles = quantiles.T - 2 # off by 2
    histogram = None
    mean = []
    var = []
    for idx in range(quantiles.shape[0]):
        row = quantiles[idx, :]
        bins = np.linspace(np.min(quantiles.flatten()), np.max(quantiles.flatten()), 100)
        hist, edges = np.histogram(row)
        if histogram is None:
            histogram = hist
        else:
            histogram = np.vstack((histogram, hist))
        mean.append(np.mean(row))
        var.append(np.var(row))

    histogram = histogram.T 
    mean = np.array(mean)
    var = np.array(var)
    np.savetxt("Mean.txt", mean)
    np.savetxt("Var.txt", var)
else:
    mean = np.loadtxt("Mean.txt")
    var = np.loadtxt("Var.txt")

quantiles = quantiles.T
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.plot(time / np.log(1e24), var)
fig.savefig("Beta100Var.pdf", bbox_inches='tight')

fig, ax = plt.subplots()
ax.set_xscale("log")
new_time = time[time >= np.log(1e24)]
new_quantiles = quantiles[time >= np.log(1e24), :]
new_mean = mean[time >= np.log(1e24)]
ax.set_xlabel(r"$t / \log(N)$")
ax.set_ylabel(r"$\mathrm{Max}^N_t - \mathrm{Mean}(\mathrm{Max}^N_t)$")
for idx in range(quantiles.shape[1]):
    print(idx)
    ax.scatter(new_time[:-1000:10] / np.log(1e24), new_quantiles[:, idx][:-1000:10] - new_mean[:-1000:10], s=0.5)

fig.savefig("Beta100Mean.pdf", bbox_inches='tight')