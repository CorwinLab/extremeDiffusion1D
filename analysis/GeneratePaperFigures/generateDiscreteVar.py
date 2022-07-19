import glob
import sys
import matplotlib

# matplotlib.rcParams['mathtext.fontset'] = 'cm'
# matplotlib.rcParams['font.family'] = 'STIXGeneral'
from matplotlib import pyplot as plt
import numpy as np
import npquad

sys.path.append("../../src")

from databases import QuartileDatabase, CDFVarianceDatabase
from quadMath import prettifyQuad
from theory import (
    theoreticalVar,
    theoreticalNthQuartVarLargeTimes,
    theoreticalNthQuartVar,
    theoreticalNthQuart,
)
from matplotlib.colors import LinearSegmentedColormap


def calculateVariance(files, verbose=True):
    first_file = np.loadtxt(files[0], delimiter=",", skiprows=1)
    time = first_file[:, 0]
    sum = 2 * first_file[:, 1]
    squared_sum = (2 * first_file[:, 1]) ** 2

    for f in files[1:]:
        data = np.loadtxt(f, delimiter=",", skiprows=1)
        sum += 2 * data[:, 1]
        squared_sum += (2 * data[:, 1]) ** 2
        if verbose:
            print(f)

    mean = sum / len(files)
    var = squared_sum / len(files) - (mean) ** 2
    return time, var, mean


def getTurnOnTime(time, var, var_thresh, N, yscale=2 / 3):
    logN = np.log(N).astype(float)
    turn_ons = var / logN ** (yscale) > var_thresh
    idx = np.where(turn_ons == 1)[0][0]
    return time[idx], var[idx]


def getLessThanT(time, mean):
    greater = mean >= time - 1
    nonzero = np.nonzero(greater)[0][-1]
    return time[nonzero]


data_dir = "/home/jacob/Desktop/corwinLabMount/CleanData/"
save_data_dir = "./Data/"
nFiles = -1
run_again = False

# Get the discrete N=1e100/300 datasets and calculate variance
discrete_300_files = glob.glob(data_dir + "MaxPart300/Quartile*.txt")[:nFiles]
max_time_300 = np.loadtxt(discrete_300_files[0], delimiter=",", skiprows=1, usecols=0)[
    -1
]

discrete_100_files = glob.glob(data_dir + "MaxPart100/Quartile*.txt")[:nFiles]
max_time_100 = np.loadtxt(discrete_100_files[0], delimiter=",", skiprows=1, usecols=0)[
    -1
]

db_300 = QuartileDatabase(discrete_300_files, nParticles=np.quad("1e300"))

db_100 = QuartileDatabase(discrete_100_files, nParticles=np.quad("1e100"))

# Get the CDF and Gumbel datasets for N=1e100/300, beta=1
gumbel_large = glob.glob(data_dir + "SweepVariance/Quartiles*.txt")[:nFiles]
max_time = np.loadtxt(gumbel_large[0], delimiter=",", skiprows=1, usecols=0)[-1]

db_gumbel = CDFVarianceDatabase(gumbel_large)

# Get the CDF and Gumbel datasets for N=2, 10, 20, beta=1
gumbel_2 = glob.glob(data_dir + "MaxPartSmall/2/Quartiles*.txt")[:nFiles]

gumbel_20 = glob.glob(data_dir + "MaxPartSmall/20/Quartiles*.txt")[:nFiles]

gumbel_11 = glob.glob(data_dir + "MaxPartSmall/7/Quartiles*.txt")[:nFiles]

# Save all the data to files or load from files
if run_again:
    db_300.calculateMeanVar(verbose=True, maxTime=max_time_300)
    db_100.calculateMeanVar(verbose=True, maxTime=max_time_100)
    db_gumbel.calculateMeanVar(verbose=True, maxTime=max_time)

    time2, var2, mean2 = calculateVariance(gumbel_2, verbose=True)
    np.savetxt(save_data_dir + "DiscreteTime2.txt", time2)
    np.savetxt(save_data_dir + "DiscreteVar2.txt", var2)
    np.savetxt(save_data_dir + "DiscreteMean2.txt", mean2)

    time20, var20, mean20 = calculateVariance(gumbel_20, verbose=True)
    np.savetxt(save_data_dir + "DiscreteTime20.txt", time20)
    np.savetxt(save_data_dir + "DiscreteVar20.txt", var20)
    np.savetxt(save_data_dir + "DiscreteMean20.txt", mean20)

    time11, var11, mean11 = calculateVariance(gumbel_11, verbose=True)
    np.savetxt(save_data_dir + "DiscreteTime11.txt", time11)
    np.savetxt(save_data_dir + "DiscreteVar11.txt", var11)
    np.savetxt(save_data_dir + "DiscreteMean11.txt", mean11)

    np.savetxt(save_data_dir + "DiscreteTime300.txt", db_300.time)
    np.savetxt(save_data_dir + "DiscreteTime100.txt", db_100.time)
    np.savetxt(save_data_dir + "Discrete300Var.txt", db_300.maxVar)
    np.savetxt(save_data_dir + "Discrete100Var.txt", db_100.maxVar)
    np.savetxt(save_data_dir + "DiscreteMean300.txt", db_300.maxMean)
    np.savetxt(save_data_dir + "DiscreteMean100.txt", db_100.maxMean)

    np.savetxt(save_data_dir + "GumbelVar.txt", db_gumbel.gumbelMean)
    np.savetxt(save_data_dir + "QuantileVar.txt", db_gumbel.var)
    np.savetxt(save_data_dir + "GumbelTimes.txt", db_gumbel.time)
else:
    db_300.time = np.loadtxt(save_data_dir + "DiscreteTime300.txt")
    db_300.maxVar = np.loadtxt(save_data_dir + "Discrete300Var.txt")
    db_300.maxMean = np.loadtxt(save_data_dir + "DiscreteMean300.txt")

    db_100.time = np.loadtxt(save_data_dir + "DiscreteTime100.txt")
    db_100.maxVar = np.loadtxt(save_data_dir + "Discrete100Var.txt")
    db_100.maxMean = np.loadtxt(save_data_dir + "DiscreteMean100.txt")

    db_gumbel.gumbelMean = np.loadtxt(save_data_dir + "GumbelVar.txt")
    db_gumbel.var = np.loadtxt(save_data_dir + "QuantileVar.txt")
    db_gumbel.time = np.loadtxt(save_data_dir + "GumbelTimes.txt")

    time20 = np.loadtxt(save_data_dir + "DiscreteTime20.txt")
    var20 = np.loadtxt(save_data_dir + "DiscreteVar20.txt")
    mean20 = np.loadtxt(save_data_dir + "DiscreteMean20.txt")

    time11 = np.loadtxt(save_data_dir + "DiscreteTime11.txt")
    var11 = np.loadtxt(save_data_dir + "DiscreteVar11.txt")
    mean11 = np.loadtxt(save_data_dir + "DiscreteMean11.txt")

    time2 = np.loadtxt(save_data_dir + "DiscreteTime2.txt")
    var2 = np.loadtxt(save_data_dir + "DiscreteVar2.txt")
    mean2 = np.loadtxt(save_data_dir + "DiscreteMean2.txt")

# Make plot of Variance as a function of time for discrete
# Look at golden ratio thing
fig, ax = plt.subplots(figsize=(3 * 1.61, 3))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t / \ln N$")
ax.set_ylabel(r"$Q(N, t) / \ln (N)^{2/3}$")
ax.set_xlim([0.5, 10 ** 4])
ax.set_ylim([10 ** -4, 10 ** 3])

number_of_plots = 5
logN300 = np.log(np.quad("1e300")).astype(float)
logN100 = np.log(np.quad("1e100")).astype(float)
logN2 = np.log(np.quad("1e2")).astype(float)
logN20 = np.log(np.quad("1e20")).astype(float)
logN11 = np.log(np.quad("1e7")).astype(float)
downsample_300 = 1
downsample_100 = 1
s = 10
alpha = 0.5
y_power = 2 / 3

cm = LinearSegmentedColormap.from_list(
    "rg", ["tab:orange", "tab:red", "tab:purple", "tab:blue"], N=256
)
colors = [cm(1.0 * i / number_of_plots / 1) for i in range(number_of_plots)]

ax.plot(
    db_300.time[::downsample_300] / logN300,
    db_300.maxVar[::downsample_300] / logN300 ** (y_power),
    ms=s,
    alpha=alpha,
    label="300",
    c=colors[0],
)
ax.plot(
    db_100.time[::downsample_100] / logN100,
    db_100.maxVar[::downsample_100] / logN100 ** (y_power),
    ms=s,
    alpha=alpha,
    label="100",
    c=colors[1],
)
ax.plot(
    time2 / logN2, var2 / logN2 ** (y_power), ms=s, alpha=alpha, label="2", c=colors[4]
)
ax.plot(
    time20 / logN20,
    var20 / logN20 ** (y_power),
    ms=s,
    alpha=alpha,
    label="20",
    c=colors[2],
)
ax.plot(
    time11 / logN11,
    var11 / logN11 ** (y_power),
    ms=s,
    alpha=alpha,
    label="10",
    c=colors[3],
)

theory300 = theoreticalVar(np.quad("1e300"), db_300.time) + db_300.time / logN300
theory100 = theoreticalVar(np.quad("1e100"), db_100.time) + db_100.time / logN100
theory20 = theoreticalVar(np.quad("1e20"), time20) + time20 / logN20
theory2 = theoreticalVar(np.quad("1e2"), time2) + time2 / logN2
theory11 = theoreticalVar(np.quad("1e7"), time11) + time11 / logN11

ax2 = fig.add_axes([0.55, 0.25, 0.3, 0.3])
ax2.set_xlabel(r"$\log_{10}N$", fontsize=8, labelpad=0)
ax2.set_ylabel(r"$T / \ln N$", fontsize=8, labelpad=0)
ax2.tick_params(axis="both", which="major", labelsize=6)

c = number_of_plots - 1
for times, vars, Nexp in zip(
    [time2, time11, time20, db_100.time, db_300.time],
    [var2, var11, var20, db_100.maxVar, db_300.maxVar],
    [2, 7, 20, 100, 300],
):
    N = np.quad(f"1e{Nexp}")
    t, v = getTurnOnTime(times, vars, 0, N)
    logN = np.log(N).astype(float)
    ax2.scatter(Nexp, t / logN, color=colors[c])
    c += -1

ax.plot(
    db_300.time / logN300,
    theory300 / logN300 ** (y_power),
    "--",
    zorder=20,
    c=colors[0],
)
ax.plot(
    db_100.time / logN100,
    theory100 / logN100 ** (y_power),
    "--",
    zorder=20,
    c=colors[1],
)
ax.plot(
    time20 / logN20,
    theory20 / logN20 ** (y_power),
    "--",
    zorder=20,
    label="theory20",
    c=colors[2],
)
ax.plot(
    time2 / logN2,
    theory2 / logN2 ** (y_power),
    "--",
    zorder=20,
    label="theory2",
    c=colors[4],
)
ax.plot(
    time11 / logN11,
    theory11 / logN11 ** (y_power),
    "--",
    zorder=20,
    label="theory11",
    c=colors[3],
)

for i in range(db_gumbel.gumbelMean.shape[1]):
    N = np.quad(db_gumbel.quantile_list[i])
    if prettifyQuad(N) == "1e100":
        continue
        # ax.scatter(db_gumbel.time[::downsample_100] / logN100, (db_gumbel.gumbelMean[:, i] + db_gumbel.var[:, i])[::downsample_100] / logN100 ** (y_power), s=s, marker='^', alpha=alpha)
    if prettifyQuad(N) == "1e300":
        continue
        # ax.scatter(db_gumbel.time[::downsample_300] / logN300, (db_gumbel.gumbelMean[:, i] + db_gumbel.var[:, i])[::downsample_300] / logN300 ** (y_power), s=s, marker='^', alpha=alpha)
# ax.legend()

fig.savefig("DiscreteVariance.png", bbox_inches="tight")

fig2, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("t / ln(N)")
ax.set_ylabel("Mean Maximum Particle")
ax.set_xlim([10 ** -2, 10 ** 4])

means = [mean2, mean11, mean20, db_100.maxMean, db_300.maxMean]
times = [time2, time11, time20, db_100.time, db_300.time]
Ns = [2, 7, 20, 100, 300]
ts = []
logNs = []
c = number_of_plots - 1
for Nexp, time, mean in zip(Ns, times, means):
    N = np.quad(f"1e{Nexp}")
    logN = np.log(N).astype(float)
    logNs.append(logN)
    theory = theoreticalNthQuart(N, time)
    ax.plot(time / logN, mean, color=colors[c])
    ax.plot(time / logN, theory, "--", color=colors[c])
    t_less = getLessThanT(time, mean)
    ts.append(t_less)
    ax.scatter(t_less / logN, t_less, color=colors[c])
    c += -1

ax3 = fig2.add_axes([0.2, 0.65, 0.2, 0.2])

ax3.plot(logNs, logNs, "--k", alpha=0.7)
ax3.scatter(logNs, ts, c=colors[::-1])
ax3.set_xscale("log")
ax3.set_yscale("log")
ax3.set_xlabel(r"$\ln(N)$", fontsize=8, labelpad=0)
ax3.set_ylabel(r"$\tau$", fontsize=8, labelpad=0)
ax3.tick_params(axis="both", which="major", labelsize=6)

fig2.savefig("DiscreteMean.png", bbox_inches="tight")
