import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os 
import json

dir = '/home/jacob/Desktop/talapasMount/JacobData/RandomDiffusionQuantile/'
mean_file = os.path.join(dir, 'Mean.txt')
var_file = os.path.join(dir, 'Var.txt')

mean = pd.read_csv(mean_file)
var = pd.read_csv(var_file)

f = open(os.path.join(dir, 'variables.json'), 'r')
vars = json.load(f)
f.close()

D0 = float(vars['D0'])
N = float(vars['N'])
sigma = float(vars['sigma'])

def theoreticalMean(D0, N, t):
    return np.sqrt(4 * D0 * np.log(N) * t)

def theoreticalVar(D0, t, N):
    return np.pi**2 / 6 * D0 * t / np.log(N)

def samplingVar(D, t, N):
    xvals = np.arange(0, 1e6, 1)
    pdf = 1 / np.sqrt(4 * np.pi * D0 * t) * np.exp(- xvals**2 / 4 / D / t)
    cdfN = (np.cumsum(pdf))**N 
    pdfN = np.diff(cdfN)
    mean = np.sum(xvals[1:] * pdfN) 
    var = np.sum(xvals[1:] ** 2 * pdfN)- mean**2
    return mean, var

s_mean, s_var = np.zeros(len(mean['Time'])), np.zeros(len(mean['Time']))

for i, t in enumerate(mean['Time'].values):
    mean_current, var_current = samplingVar(D0, t, N)
    s_mean[i] = mean_current
    s_var[i] = var_current
    print(mean_current)
    print(i)

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.plot(mean['Time'], mean['Mean Mean'])
ax.plot(mean['Time'], theoreticalMean(D0, N, mean['Time']), ls='--', c='k')
ax.plot(mean['Time'], s_mean, ls='--', c='m')
fig.savefig("Mean.png")

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.plot(mean['Time'], mean['Mean Var'])
ax.plot(mean['Time'], theoreticalVar(D0, mean['Time'], N), ls='--', c='k')
fig.savefig("Var.png")

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.plot(var['Time'], var['Var Mean'])
xvals = np.array([10, 10**3])
ax.plot(xvals, xvals ** (-1/2) / 10000 / 2)
fig.savefig("VarMean.png")

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.plot(mean['Time'], (mean['Mean Var'] - theoreticalVar(D0, mean['Time'], N)) / mean['Time'])

fig.savefig("VarDiff.png")