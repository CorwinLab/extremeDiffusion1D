import numpy as np 
import npquad
import os
import glob
import sys 
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.special import digamma

sys.path.append("../../dataAnalysis")
from overalldatabase import Database
from theory import quantileVar, quantileVarLongTimeBetaDist, log_moving_average, log_moving_average_error

db = Database()
beta_dir = "/home/jacob/Desktop/corwinLabMount/CleanData/CDFBetaSweep/"
dirs = os.listdir(beta_dir)
for dir in dirs:
    path = os.path.join(beta_dir, dir)
    beta = float(dir.split("/")[-1])
    db.add_directory(path, dir_type="Gumbel")

betas = db.betas()
Nexp = 24
N = 1e24
logN = np.log(np.quad(f"1e24")).astype(float)
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("t/log(N)")
ax.set_ylabel("Var(Env)")
cm = LinearSegmentedColormap.from_list(
    "rg", ["tab:orange", "tab:red", "tab:purple", "tab:blue"], N=256
)
colors = [cm(1.0 * i / len(betas) / 1) for i in range(len(betas))]
alpha=0.5
ax.set_xlim([3*10**-1, 5000])
ax.set_ylim([1*10**-3, 10**5])
for i, b in enumerate(betas): 
    db_new = db.getBetas(b)
    cdf_df, max_df = db_new.getMeanVarN(Nexp)
    time = cdf_df['time'].values
    var = cdf_df['Var Quantile'].values
    b = float(b)
    ax.plot(time / logN, var, c=colors[i], alpha=alpha, label=fr"$\beta={b}$")
    if b == 1: 
        ax.plot(time / logN, quantileVar(N, time), ls='--', c=colors[i])
    if b < 1: 
        time = np.loadtxt(f"./Theory/times{b}.txt")
        var = np.loadtxt(f"./Theory/variance{b}.txt")
        var[time <= 0.1 + logN / (digamma(2 * b) - digamma(b))] = 0
        ax.plot(time / logN, var, ls='--', c=colors[i])
    if (b > 1):
        var = quantileVarLongTimeBetaDist(N, time, b)
        var[time <= logN / (digamma(2 * b) - digamma(b))] = 0
        ax.plot(time / logN, var, ls='--', c=colors[i])

    xval = np.log(N) **(3/2) * (digamma(2 * b) - digamma(b))**(3/2)
    print(xval)
    ax.vlines(xval / np.log(N), 10**-3, 10**5, color=colors[i])

end_coord = (100, 10**-2)
start_coord = (20, 3*10**3)
ax.annotate(
    "",
    xy=end_coord,
    xytext=start_coord,
    arrowprops=dict(
        shrink=0.0,
        facecolor="gray",
        edgecolor="white",
        width=50,
        headwidth=100,
        headlength=50,
        alpha=0.3,
    ),
    zorder=0,
)

leg = ax.legend(
    loc="upper left",
    framealpha=0,
    labelcolor=colors,
    handlelength=0,
    handletextpad=0,
)
for item in leg.legendHandles:
    item.set_visible(False)

fig.savefig("QuantileVariance.pdf", bbox_inches='tight')

