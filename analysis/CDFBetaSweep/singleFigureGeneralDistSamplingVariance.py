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
from theory import quantileVarLongTimeBetaDist

Nexp = 24
N = 1e24
logN = np.log(np.quad(f"1e24")).astype(float)
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t/\log(N)$")
ax.set_ylabel(r"$\mathrm{Var}(\mathrm{Sam}_t^N)$")
ax.set_xlim([3*10**-1, 5000])
ax.set_ylim([1*10**-3, 10**4])

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

alpha=0.5

for i, b in enumerate(betas): 
    db_new = db.getBetas(b)
    cdf_df, max_df = db_new.getMeanVarN(Nexp)
    time = cdf_df['time'].values
    var = cdf_df['Gumbel Mean Variance'].values

    b = float(b)
    ax.plot(time / logN, var, c=beta_color, alpha=alpha)
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
    ax.plot(cdf_df['time'] / logN, cdf_df['Gumbel Mean Variance'], alpha=alpha, c=bates_color)

# Plot delta variance
delta_dir = "/home/jacob/Desktop/talapasMount/JacobData/Delta"
dirs = os.listdir(delta_dir)
for dir in dirs:
    db = Database()
    path = os.path.join(delta_dir, dir)
    beta = float(dir.split("/")[-1])
    db.add_directory(path, dir_type="Gumbel")
    cdf_df, max_df = db.getMeanVarN(Nexp)
    ax.plot(cdf_df['time'] / logN, cdf_df['Gumbel Mean Variance'], alpha=alpha, c=delta_color)

# Plot quadratic variance
quadratic_dir = "/home/jacob/Desktop/talapasMount/JacobData/Quadratic"
dirs = os.listdir(quadratic_dir)
for dir in dirs:
    db = Database()
    path = os.path.join(quadratic_dir, dir)
    beta = float(dir.split("/")[-1])
    db.add_directory(path, dir_type="Gumbel")
    cdf_df, max_df = db.getMeanVarN(Nexp)
    ax.plot(cdf_df['time'] / logN, cdf_df['Gumbel Mean Variance'], alpha=alpha, c=quad_color)

# Plot uniform variance
uniform_dir = "/home/jacob/Desktop/talapasMount/JacobData/Uniform"
dirs = os.listdir(uniform_dir)
for dir in dirs:
    db = Database()
    path = os.path.join(uniform_dir, dir)
    beta = float(dir.split("/")[-1])
    db.add_directory(path, dir_type="Gumbel")
    cdf_df, max_df = db.getMeanVarN(Nexp)
    ax.plot(cdf_df['time'] / logN, cdf_df['Gumbel Mean Variance'], alpha=alpha, c=uniform_color)

labels = ['Beta', 'Delta', 'Comp. Triangular', 'Quadratic', 'Bates', 'Uniform']
colors = [beta_color, delta_color, inv_triang_color, quad_color, bates_color, uniform_color]

leg = ax.legend(
    loc="upper left",
    labels=labels,
    framealpha=0,
    labelcolor=colors,
    handlelength=0,
    handletextpad=0,
)
for item in leg.legendHandles:
    item.set_visible(False)

fig.savefig("GeneralSamplingVariance.pdf", bbox_inches='tight')

