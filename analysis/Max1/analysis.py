import sys
sys.path.append("../../src")

from overalldatabase import Database
from matplotlib import pyplot as plt
import theory
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import npquad
import json
import os

db = Database()
path = '/home/jacob/Desktop/talapasMount/JacobData/Max1/'
db.add_directory(path, dir_type='Max')
#db.calculateMeanVar(path, verbose=True)

for dir in db.dirs.keys():
    f = open(os.path.join(dir, 'analysis.json'), 'r')
    x = json.load(f)
    print(dir, ' Systems:', x['number_of_systems'])

cdf_df, max_df = db.getMeanVarN(1)
N = np.quad("1e1")
logN = np.log(10).astype(float)
max_df['Var Max'] = max_df['Var Max'] * 4

var_theory = theory.quantileVar(N, max_df['time'].values, crossover=logN**(1.5), width=logN**(4/3))
gumbel_theory = theory.gumbel_var(max_df['time'].values, N)

w = 1
env_recovered = max_df['Var Max'] - gumbel_theory
env_recovered = np.convolve(env_recovered, np.ones(w), mode='valid') / w
env_time = np.convolve(max_df['time'].values, np.ones(w), mode='valid') / w

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylabel("Var(max)")
ax.set_xlabel("t / logN")

ax.plot(max_df['time'] / logN, max_df['Var Max'], c='r', alpha=0.7)
ax.plot(max_df['time'] / logN, var_theory + gumbel_theory, '--', c='r')
ax.plot(max_df['time'] / logN, gumbel_theory, '--', c='m')

ax.plot(max_df['time'] / logN, var_theory, c='b')
ax.plot(env_time / logN, env_recovered, c='tab:orange')

fig.savefig("Max1.png", bbox_inches='tight')

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.plot(env_time / logN, env_recovered, c='tab:orange')
fig.savefig("Residual1.png", bbox_inches='tight')
