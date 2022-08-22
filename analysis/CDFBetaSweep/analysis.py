import numpy as np 
import npquad
import os
import glob
import sys 
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.append("../../dataAnalysis")
from overalldatabase import Database

db = Database()
beta_dir = "/home/jacob/Desktop/corwinLabMount/CleanData/CDFBetaSweep/"
dirs = os.listdir(beta_dir)
for dir in dirs:
    path = os.path.join(beta_dir, dir)
    beta = float(dir.split("/")[-1])
    db.add_directory(path, dir_type="Gumbel")
    #db.calculateMeanVar(path, verbose=True, maxTime=276310)

betas = db.betas()
N=24
logN = np.log(np.quad(f"1e24")).astype(float)

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("t/log(N)")
ax.set_ylabel("Var(Env)")
ax.set_xlim([1 / logN, 5000])
cm = LinearSegmentedColormap.from_list(
    "rg", ["tab:orange", "tab:red", "tab:purple", "tab:blue"], N=256
)
colors = [cm(1.0 * i / len(betas) / 1) for i in range(len(betas))]
alpha=0.75

for i, b in enumerate(betas): 
    db_new = db.getBetas(b)
    cdf_df, max_df = db_new.getMeanVarN(N)
    time = cdf_df['time'].values
    var = cdf_df['Var Quantile'].values
    ax.plot(time / logN, var, c=colors[i], alpha=alpha)

time = np.loadtxt("times.txt")
var01 = np.loadtxt("Variance01.txt")
var001 = np.loadtxt("Variance001.txt")
ax.plot(time / logN, var01, ls='--', c='b')
ax.plot(time / logN, var001, ls='--', c='r')

fig.savefig("QuantileVariance.pdf", bbox_inches='tight')

