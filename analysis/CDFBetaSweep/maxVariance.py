import numpy as np 
import os
import sys 
from matplotlib import pyplot as plt
from scipy.special import digamma
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

sys.path.append("../../dataAnalysis")
from overalldatabase import Database
from theory import quantileVar, gumbel_var

db = Database()
beta_dir = "/home/jacob/Desktop/talapasMount/JacobData/BetaSweepPaper/"
folders = os.listdir(beta_dir)
for f in folders:
    db.add_directory(os.path.join(beta_dir, f), dir_type="Max")

betas = db.betas()
betas = [float(b) for b in betas]
betas = sorted(betas)
Nexp = 24
N = 1e24
logN = np.log(np.quad(f"1e24")).astype(float)
cm = LinearSegmentedColormap.from_list(
    "rg", ["tab:orange", "tab:red", "tab:purple", "tab:blue"], N=256
)
colors = [cm(1.0 * i / len(betas) / 1) for i in range(len(betas))]

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t / \log(N)$")
ax.set_ylabel(r"$\mathrm{Var}(\mathrm{Max}^N_t)$")
ax.set_xlim([0.1, 5000])
ax.set_ylim([10**-2, 10**5])

for i, b in enumerate(betas): 
    db_new = db.getBetas(b)
    _, max_df = db_new.getMeanVarN(Nexp)
    max_df['Mean Max'] *= 2 
    max_df['Var Max'] *= 4
    ax.plot(max_df['time'] / logN, max_df['Var Max'], c=colors[i], alpha=0.75, label=fr'$\beta={b}$')

    if b == 1:
        ax.plot(max_df['time'] / logN, quantileVar(N, max_df['time'].values) + gumbel_var(max_df['time'].values, N), c=colors[i], ls='--')

    if b > 1:
        b = int(b)

    env_var_file = f"./Theory/EnvironmentalVariance{b}.txt"
    sam_var_file = f"./Theory/SamplingVariance{b}.txt"
    if os.path.exists(env_var_file) and os.path.exists(sam_var_file):
        env_var = pd.read_csv(env_var_file, header=None)
        env_var.replace("Indeterminate", np.nan, inplace=True)
        env_var = env_var.values.astype(float)
        sampling_variance = pd.read_csv(sam_var_file, header=None)
        sampling_variance.replace("Indeterminate", np.nan, inplace=True)
        sam_var = sampling_variance.values.astype(float)
        time = np.loadtxt(f"./Theory/times{b}.txt")

        ax.plot(time / logN, env_var + sam_var, c=colors[i], ls='--')

leg = ax.legend(
    loc="upper left",
    framealpha=0,
    labelcolor=colors,
    handlelength=0,
    handletextpad=0,
)
for item in leg.legendHandles:
    item.set_visible(False)

fig.savefig("MaxVariance.pdf", bbox_inches='tight')