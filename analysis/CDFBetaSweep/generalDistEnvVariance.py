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

Nexp = 24
N = 1e24
logN = np.log(np.quad(f"1e24")).astype(float)
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("t/log(N)")
ax.set_ylabel("Var(Env)")
ax.set_xlim([3*10**-1, 5000])
ax.set_ylim([1*10**-3, 10**5])

# Plot beta variance
db = Database()
beta_dir = "/home/jacob/Desktop/corwinLabMount/CleanData/CDFBetaSweep/"
dirs = os.listdir(beta_dir)
for dir in dirs:
    path = os.path.join(beta_dir, dir)
    beta = float(dir.split("/")[-1])
    db.add_directory(path, dir_type="Gumbel")

betas = db.betas()

delta_color = 'tab:blue'
beta_color = 'tab:red'
inv_triang_color = 'tab:purple'
quad_color = 'tab:orange'
bates_color = 'tab:pink'
uniform_color = 'tab:cyan'

alpha=0.75

for i, b in enumerate(betas): 
    db_new = db.getBetas(b)
    cdf_df, max_df = db_new.getMeanVarN(Nexp)
    time = cdf_df['time'].values
    var = cdf_df['Var Quantile'].values

    b = float(b)
    ax.plot(time / logN, var, c=beta_color, alpha=alpha, label=fr"$\beta={b}$")
    var = quantileVarLongTimeBetaDist(N, time, b)
    var[time <= 0.1 + logN / (digamma(2 * b) - digamma(b))] = 0
    #ax.plot(time / logN, var, ls='--', c=colors[i])

# Plot bates variance
bates_dir = "/home/jacob/Desktop/talapasMount/JacobData/Bates"
dirs = os.listdir(bates_dir)
for dir in dirs:
    db = Database()
    path = os.path.join(bates_dir, dir)
    beta = float(dir.split("/")[-1])
    db.add_directory(path, dir_type="Gumbel")
    cdf_df, max_df = db.getMeanVarN(Nexp)
    ax.plot(cdf_df['time'] / logN, cdf_df['Var Quantile'], alpha=alpha, c=bates_color)

# Plot delta variance
delta_dir = "/home/jacob/Desktop/talapasMount/JacobData/Delta"
dirs = os.listdir(delta_dir)
for dir in dirs:
    db = Database()
    path = os.path.join(delta_dir, dir)
    beta = float(dir.split("/")[-1])
    db.add_directory(path, dir_type="Gumbel")
    cdf_df, max_df = db.getMeanVarN(Nexp)
    ax.plot(cdf_df['time'] / logN, cdf_df['Var Quantile'], alpha=alpha, c=delta_color)

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

'''
leg = ax.legend(
    loc="upper left",
    framealpha=0,
    labelcolor=colors,
    handlelength=0,
    handletextpad=0,
)
for item in leg.legendHandles:
    item.set_visible(False)
'''

fig.savefig("GeneralQuantileVariance.pdf", bbox_inches='tight')

