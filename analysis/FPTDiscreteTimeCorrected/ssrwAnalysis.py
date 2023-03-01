import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import sys 
sys.path.append("../../dataAnalysis")

from numericalFPT import getNParticleMeanVar
from theory import log_moving_average, log_moving_average_error

einstein_df = pd.read_csv('/home/jacob/Desktop/talapasMount/JacobData/FPTDiscreteE/12/MeanVariance.csv')

N = 1e12
logN = np.log(N)
xvals = einstein_df['Distance'].values
mean, sampling_variance = getNParticleMeanVar(xvals, N)

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.plot(xvals, einstein_df['Variance'])
ax.plot(xvals, sampling_variance, ls='--')
fig.savefig("SSRWVariance.png", bbox_inches='tight')

dist_new, env_var = log_moving_average(xvals, einstein_df['Variance'] - sampling_variance, 10 ** (1/5))

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("symlog")
ax.scatter(dist_new, env_var)
fig.savefig("SSRWVarianceResidual.png", bbox_inches='tight')