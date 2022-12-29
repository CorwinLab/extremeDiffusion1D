import numpy as np 
import npquad
import os
import glob
import sys 
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.special import digamma
import pandas as pd

sys.path.append("../../dataAnalysis")
from overalldatabase import Database
from theory import gumbel_var

db = Database()
beta_dir = "/home/jacob/Desktop/corwinLabMount/CleanData/CDFBetaSweep/"
dirs = os.listdir(beta_dir)
for dir in dirs:
    path = os.path.join(beta_dir, dir)
    beta = float(dir.split("/")[-1])
    db.add_directory(path, dir_type="Gumbel")
    #db.calculateMeanVar(path, verbose=True, maxTime=276310)

betas = db.betas()
Nexp = 24
N = float(f"1e{Nexp}")
logN = np.log(N).astype(float)
print(betas)
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t/\log(N)$")
ax.set_ylabel(r"$\mathrm{Var}(\mathrm{Sam}_t^N)$")
ax.set_xlim([0.1, 5000])
ax.set_ylim([10**-3, 10**4])
cm = LinearSegmentedColormap.from_list(
    "rg", ["tab:orange", "tab:red", "tab:purple", "tab:blue"], N=256
)
colors = [cm(1.0 * i / len(betas) / 1) for i in range(len(betas))]
alpha=0.75

for i, b in enumerate(betas): 
    db_new = db.getBetas(b)
    cdf_df, max_df = db_new.getMeanVarN(Nexp)
    time = cdf_df['time'].values
    var = cdf_df['Gumbel Mean Variance'].values
    ax.plot(time / logN, var, c=colors[i], alpha=alpha, label=fr'$\beta={b}$')

    if float(b) == 1:
        ax.plot(time / logN, gumbel_var(time, N), c=colors[i], ls='--')
        continue
    if b > 1:
        b = int(b)
    sampling_variance = pd.read_csv(f"./Theory/SamplingVariance{b}.txt")
    sampling_variance.replace("Indeterminate", np.nan, inplace=True)
    time = pd.read_csv(f"./Theory/times{b}.txt")
    sampling_variance = sampling_variance.values.astype(float)
    ax.plot(time / logN, sampling_variance, c=colors[i], ls='--')

leg = ax.legend(
    loc="upper left",
    framealpha=0,
    labelcolor=colors,
    handlelength=0,
    handletextpad=0,
)
for item in leg.legendHandles:
    item.set_visible(False)

fig.savefig("SamplingVariance.pdf", bbox_inches='tight')

