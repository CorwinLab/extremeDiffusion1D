import glob
import sys
import matplotlib

# matplotlib.rcParams['mathtext.fontset'] = 'cm'
# matplotlib.rcParams['font.family'] = 'STIXGeneral'
from matplotlib import pyplot as plt
import numpy as np
import npquad

sys.path.append("../../src")

from databases import QuartileDatabase, CDFVarianceDatabase, CDFQuartileDatabase
from quadMath import prettifyQuad
from theory import theoreticalNthQuartVar, theoreticalNthQuartVarLargeTimes
from matplotlib.colors import LinearSegmentedColormap

data_dir = "/home/jacob/Desktop/corwinLabMount/CleanData/"
save_data_dir = "./Data/"
nFiles = 10
run_again = True
number_of_plots = 5

sweep_variance_file = data_dir + "SweepVariance/Quartiles0.txt"
quantile_vals = []
with open(sweep_variance_file, "r") as f:
    header = f.readline()
    header = header.split(",")
    for elem in header[1:]:
        if "var" in elem:
            continue
        else:
            quantile_vals.append(np.quad(elem))

fig, ax = plt.subplots(figsize=(3 * 1.61, 3))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t / \ln N$")
ax.set_ylabel(r"$Q(N, t) / \ln (N)^{2/3}$")
ax.set_xlim([0.5, 10 ** 4])

cm = LinearSegmentedColormap.from_list(
    "rg", ["tab:orange", "tab:red", "tab:purple", "tab:blue"], N=256
)
colors = [cm(1.0 * i / number_of_plots / 1) for i in range(number_of_plots)]
alpha = 0.5
# Plot the N=1e100, 1e300 data
quantiles = np.loadtxt("./Data/QuantileVar.txt")
time = np.loadtxt("./Data/GumbelTimes.txt")
for i in range(quantiles.shape[1]):
    var = quantiles[:, i]
    N = quantile_vals[i]
    if prettifyQuad(N) == "1e100":
        ax.plot(
            time / np.log(N), var / (np.log(N) ** (2 / 3)), alpha=alpha, c=colors[1]
        )
    elif prettifyQuad(N) == "1e300":
        ax.plot(
            time / np.log(N), var / (np.log(N) ** (2 / 3)), alpha=alpha, c=colors[0]
        )

N = quantile_vals[0]
crossover = np.log(N).astype(float) * 10 ** 2
width = crossover
theory = theoreticalVar(N, time, crossover, width)
ax.plot(time / np.log(N), theory / (np.log(N) ** (2 / 3)), "--k", zorder=20)

# Plot the N=1e2, 1e7, 1e10 data
files = glob.glob(data_dir + "CDFSmallBeta1/Q*.txt")
db = CDFVarianceDatabase(files)
db.calculateMeanVar(verbose=True)

for i in range(db.var.shape[1]):
    N = np.quad(db.quantile_list[i])
    logN = np.log(N).astype(float)
    if N == np.quad("1e2"):
        ax.plot(
            db.time / logN, db.var[:, i] / logN ** (2 / 3), alpha=alpha, c=colors[4]
        )
    elif N == np.quad("1e11"):
        ax.plot(
            db.time / logN, db.var[:, i] / logN ** (2 / 3), alpha=alpha, c=colors[3]
        )
    elif N == np.quad("1e20"):
        ax.plot(
            db.time / logN, db.var[:, i] / logN ** (2 / 3), alpha=alpha, c=colors[2]
        )

fig.savefig("PDFVariance.png", bbox_inches="tight")
