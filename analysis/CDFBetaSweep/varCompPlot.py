import numpy as np 
import os
import sys 
from matplotlib import pyplot as plt
from scipy.special import digamma
import pandas as pd

sys.path.append("../../dataAnalysis")
from overalldatabase import Database

db = Database()
beta_dir = "/home/jacob/Desktop/talapasMount/JacobData/BetaSweepPaper/0.1"
db.add_directory(beta_dir, dir_type="Max")
#db.calculateMeanVar(beta_dir, verbose=True, maxTime=276310)

beta_dir = "/home/jacob/Desktop/corwinLabMount/CleanData/CDFBetaSweep/"
dirs = os.listdir(beta_dir)
for dir in dirs:
    path = os.path.join(beta_dir, dir)
    beta = float(dir.split("/")[-1])
    db.add_directory(path, dir_type="Gumbel")

b = 0.1
Nexp = 24
N = float(f"1e{Nexp}")
logN = np.log(N)

max_color = "tab:red"
quantile_color = "tab:blue"
gumbel_color = "tab:green"
alpha = 0.75

db_new = db.getBetas(b)
cdf_df, max_df = db_new.getMeanVarN(Nexp)
max_df['Var Max'] *= 4
max_df['Mean Max'] *= 2

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t/\log(N)$")
ax.set_ylabel(r"$\mathrm{Var}$")
ax.set_xlim([2 * 10**-2, 5000])
ax.set_ylim([10**-3, 2*10**4])
ax.set_title(fr"$\beta={b}$")
ax.plot(cdf_df['time'] / logN, cdf_df['Var Quantile'], label=r'$\mathrm{Var}(\mathrm{Env}^N_t)$', color=quantile_color, alpha=alpha)
ax.plot(cdf_df['time'] / logN, cdf_df['Gumbel Mean Variance'], label=r'$\mathrm{Var}(\mathrm{Sam}^N_t)$', color=gumbel_color, alpha=alpha)
ax.plot(max_df['time'] / logN, max_df['Var Max'], label=r'$\mathrm{Var}(\mathrm{Max}^N_t)$', color=max_color, alpha=alpha)

time = np.loadtxt(f"./Theory/times{b}.txt")
var = np.loadtxt(f"./Theory/variance{b}.txt")
var[time <= 0.1 + logN / (digamma(2 * b) - digamma(b))] = 0
ax.plot(time / logN, var, ls='--', c=quantile_color)

sampling_variance = pd.read_csv(f"./Theory/SamplingVariance{b}.txt")
sampling_variance.replace("Indeterminate", np.nan, inplace=True)
time = pd.read_csv(f"./Theory/times{b}.txt")
sampling_variance = sampling_variance.values.astype(float)
ax.plot(time / logN, sampling_variance, c=gumbel_color, ls='--')

ax.plot(cdf_df['time'] / logN, cdf_df['Var Quantile'] + cdf_df['Gumbel Mean Variance'], label=r'$\mathrm{Var}(\mathrm{Sam}^N_t) + \mathrm{Var}(\mathrm{Env}^N_t)$', c='tab:purple', alpha=alpha)

leg = ax.legend(
    loc="upper left",
    framealpha=0,
    labelcolor=[quantile_color, gumbel_color, max_color, 'tab:purple'],
    handlelength=0,
    handletextpad=0,
)
for item in leg.legendHandles:
    item.set_visible(False)

fig.savefig("MaxQuantPlot.pdf", bbox_inches='tight')