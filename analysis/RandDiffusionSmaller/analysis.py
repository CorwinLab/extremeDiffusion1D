import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
import os 
import json 
import sys 
sys.path.append("../../dataAnalysis")
from theory import KPZ_var_fit

dir = '/home/jacob/Desktop/talapasMount/JacobData/RandomDiffusionSmaller/'
f = open(os.path.join(dir, 'variables.json'))
vars = json.load(f)
f.close()

D0 = float(vars['D0'])
v = float(vars['v'])
sigma = float(vars['sigma'])
def theoreticalMean(v, D0, t):
    return -v**2 * t / 4 / D0

def theoreticalVar(sigma, v, D0, t):
    tstar = 2 *(2*D0)**9 / sigma**4 / v**8
    print(D0, sigma, v)
    return KPZ_var_fit(t / tstar)

mean_file = os.path.join(dir, 'Mean.txt')
var_file = os.path.join(dir, 'Var.txt')

mean = pd.read_csv(mean_file)
var = pd.read_csv(var_file)

# Plot Mean Probability
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylabel(r"$-\mathrm{Mean}(\log(P(X=vt^{3/4}, t)))$")
ax.set_xlabel("t / log(N)")
ax.plot(mean['Time'], -mean['Probability'])
ax.plot(mean['Time'], -theoreticalMean(v, D0, mean['Time']), ls='--', c='k', label=r'$\frac{v^2}{4 D_0} t$')
#ax.set_ylim([1, 2*10**2])
ax.set_xlim([min(mean['Time']), max(mean['Time'])])
ax.legend()
fig.savefig("ProbabilityMean.pdf", bbox_inches='tight')

# Plot Var Quantile
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylabel(r"$\mathrm{Var}(\log(P(X=vt^{3/4}, t)))$")
ax.set_xlabel("t")
ax.plot(var['Time'], var['Probability'])
xvals = np.array([10**2, 10**3])
ax.plot(xvals, xvals **(1/2) / 1000, ls='--')
ax.plot(var['Time'], theoreticalVar(sigma, v, D0, var['Time'].to_numpy()) / 100)
fig.savefig("ProbabilityVar.pdf", bbox_inches='tight')