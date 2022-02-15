import sys
sys.path.append("../../src")

from overalldatabase import Database
from matplotlib import pyplot as plt
import theory
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import npquad

def getLessThanT(time, mean):
    greater = mean >= time-1
    nonzero = np.nonzero(greater)[0][-1]
    return time[nonzero]

db = Database()
db.add_directory('/home/jacob/Desktop/corwinLabMount/CleanData/MaxBetaSweep2/1/', dir_type='Max')
db.add_directory('/home/jacob/Desktop/corwinLabMount/CleanData/MaxBetaSweep2/5/', dir_type='Max')
db.add_directory('/home/jacob/Desktop/corwinLabMount/CleanData/SweepVariance/', dir_type='Gumbel')
db.add_directory('/home/jacob/Desktop/corwinLabMount/CleanData/MaxPart100/', dir_type='Max')
db.add_directory('/home/jacob/Desktop/corwinLabMount/CleanData/MaxPart300/', dir_type='Max')
db.add_directory('/home/jacob/Desktop/corwinLabMount/CleanData/MaxPartSmall/2/', dir_type='Max')
db.add_directory('/home/jacob/Desktop/corwinLabMount/CleanData/MaxPartSmall/6/', dir_type='Max')
db.add_directory('/home/jacob/Desktop/corwinLabMount/CleanData/MaxPartSmall/20/', dir_type='Max')
db.add_directory('/home/jacob/Desktop/corwinLabMount/CleanData/CDFSmallBeta1/', dir_type='Gumbel')

#db.calculateMeanVar('/home/jacob/Desktop/corwinLabMount/CleanData/CDFSmallBeta1/', verbose=True)
db = db.getBetas(1)

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t / \ln(N)$")
ax.set_ylabel(r"$\mathrm{Var}(Env_{t}^{(N)})$")

quantiles = [2, 6, 20, 100, 300]
cm = LinearSegmentedColormap.from_list('rg', ['tab:orange', 'tab:red', "tab:purple", 'tab:blue'], N=256)
colors = [
    cm(1.0 * i / len(quantiles) / 1) for i in range(len(quantiles))
]
ypower = 0
for i, N in enumerate(quantiles[-2:]):
    cdf_df, max_df = db.getMeanVarN(N)

    Nquad = np.quad(f"1e{N}")
    logN = np.log(Nquad).astype(float)
    short_time = np.piecewise(cdf_df['time'].values,
                                [cdf_df['time'].values <= logN, cdf_df['time'].values > logN],
                                [lambda t: np.nan, lambda t: theory.quantileVarShortTime(Nquad, t)])
    long_time = theory.quantileVarLongTime(Nquad, cdf_df['time'].values)
    var = theory.quantileVar(Nquad, cdf_df['time'].values, crossover=logN**(1.5), width=logN**(4/3))

    ax.plot(cdf_df['time'] / logN, cdf_df['Var Quantile'] / logN**ypower, alpha=0.5, c=colors[i])
    #ax.plot(cdf_df['time'] / logN, predicted / logN**ypower, ls='--', c=colors[i])
    ax.plot(cdf_df['time'] / logN, long_time / logN**ypower, ls=':', c=colors[i])
    ax.plot(cdf_df['time'] / logN, short_time / logN**ypower, ls='--', c=colors[i])
    ax.plot(cdf_df['time'] / logN, var / logN**ypower, ls='-.', c='k', zorder=0)

ax.set_xlim([0.5, 5*10**3])
ax.set_ylim([10**-1, 3*10**3])
ax.grid(True)
fig.savefig("QuantileVar.png")

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t / \ln(N)$")
ax.set_ylabel(r"$\bar{X}_Q(N, t)$")

fig2, ax2 = plt.subplots()
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel(r"$t / \ln(N)$")
ax2.set_ylabel(r"$Residual$")

quantiles = [2, 6, 20, 100, 300]
cm = LinearSegmentedColormap.from_list('rg', ['tab:orange', 'tab:red', "tab:purple", 'tab:blue'], N=256)
colors = [
    cm(1.0 * i / len(quantiles) / 1) for i in range(len(quantiles))
]
ypower = 2/3
for i, N in enumerate(quantiles):
    cdf_df, max_df = db.getMeanVarN(N)
    Nquad = np.quad(f"1e{N}")
    logN = np.log(Nquad).astype(float)
    predicted = theory.quantileMean(Nquad, (cdf_df['time']).values)
    ax.plot((cdf_df['time']) / logN, cdf_df['Mean Quantile'] - 2, alpha=0.5, c=colors[i])
    ax.plot((cdf_df['time']) / logN, predicted, ls='--', c=colors[i])
    ax2.plot(cdf_df['time'] / logN, predicted - cdf_df['Mean Quantile'], c=colors[i])

ax.set_xlim([10**-3, 3*10**3])
ax.set_ylim([1, 7*10**4])
fig.savefig("QuantileMean.png")
ax2.set_xlim([1, 10**4])
ax2.set_ylim([10**-2, 2 * 10**2])
fig2.savefig("ResidualMean.png")

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t / \ln(N)$")
ax.set_ylabel(r"$\sigma^{2}_{max} (N, t) / \ln(N)^{2/3}$")
quantiles = [2, 6, 20, 100, 300]
cm = LinearSegmentedColormap.from_list('rg', ['tab:orange', 'tab:red', "tab:purple", 'tab:blue'], N=256)
colors = [
    cm(1.0 * i / len(quantiles) / 1) for i in range(len(quantiles))
]

for i, quantile in enumerate(quantiles):
    N = np.quad(f"1e{quantile}")
    logN = np.log(N).astype(float)
    cdf_df, max_df = db.getMeanVarN(quantile)

    if quantile != 100:
        max_df['Mean Max'] = max_df['Mean Max'] * 2
    else:
        max_df['Mean Max'] = (max_df['Mean Max'] - (max_df['time'] / 2))
    max_df['Var Max'] = max_df['Var Max'] * 4

    var_theory = theory.quantileVar(N, max_df['time'].values)

    ax.plot(max_df['time'] / logN, (var_theory)/ logN**(2/3) + np.pi**2 / 12 * max_df['time'] / logN / logN**(2/3) , '--', c=colors[i])
    ax.plot(max_df['time'] / logN, max_df['Var Max'] / logN**(2/3), label=quantile, c=colors[i], alpha=0.5)

end_coord = (200, 0.5)
start_coord = (100, 2*10**2)
dx = np.log(start_coord[0]) - np.log(end_coord[0])
dy = np.log(start_coord[1]) - np.log(end_coord[1])
theta = np.rad2deg(np.arctan2(dy, dx))
ax.annotate("", xy=end_coord, xytext=start_coord,
        arrowprops=dict(shrink=0., facecolor='gray', edgecolor='white', width=20, headwidth=50, headlength=30, alpha=0.5), zorder=0)
ax.annotate(r"$N=10^{2}$", xy=(50, 3 *10**2), c=colors[0], rotation=-(90-theta), rotation_mode='anchor')

ax.set_xlim([0.3, 5*10**3])
ax.set_ylim([10**-4, 10**3])
fig.savefig("Var.png")

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t / \ln(N)$")
ax.set_ylabel(r"$\overline{X_{max}}(N, t)$")

ax2 = fig.add_axes([0.2, 0.57, 0.25, 0.25])
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel(r"$\ln(N)$", fontsize=8, labelpad=0)
ax2.set_ylabel(r"$\tau$", fontsize=8, labelpad=0)
ax2.tick_params(axis='both', which='major', labelsize=6)

quantiles = [2, 6, 20, 100, 300]
cm = LinearSegmentedColormap.from_list('rg', ['tab:orange', 'tab:red', "tab:purple", 'tab:blue'], N=256)
colors = [
    cm(1.0 * i / len(quantiles) / 1) for i in range(len(quantiles))
]
logNs = []
t_less_than = []

for i, quantile in enumerate(quantiles):
    N = np.quad(f"1e{quantile}")
    logN = np.log(N).astype(float)
    cdf_df, max_df = db.getMeanVarN(quantile)

    if quantile != 100:
        max_df['Mean Max'] = max_df['Mean Max'] * 2
    else:
        max_df['Mean Max'] = 2*(max_df['Mean Max'] - (max_df['time'] / 2))

    max_df['Var Max'] = max_df['Var Max'] * 4

    var_theory = theory.quantileMean(N, max_df['time'].values)

    ax.plot(max_df['time'] / logN, max_df['Mean Max'], label=quantile, c=colors[i], alpha=0.8)
    ax.plot(max_df['time'] / logN, var_theory, '--', c=colors[i])
    ax.plot(max_df['time'] / logN, np.sqrt(2 * max_df['time'] * logN), c=colors[i], alpha=0.8, ls='dotted')

    t_less_than.append(getLessThanT(max_df['time'].values, max_df['Mean Max'].values))
    logNs.append(logN)

start_coord = (200, 30)
end_coord = (8, 2*10**4)
dx = np.log(start_coord[0]) - np.log(end_coord[0])
dy = np.log(start_coord[1]) - np.log(end_coord[1])
theta = np.rad2deg(np.arctan2(dy, dx))
ax.annotate("", xy=end_coord, xytext=start_coord,
        arrowprops=dict(shrink=0., facecolor='gray', edgecolor='white', width=50, headwidth=100, headlength=60, alpha=0.5), zorder=0)
ax.annotate(r"$N=10^{2}$", xy=(start_coord[0] - 90, start_coord[1] - 17), c=colors[0], rotation=90-abs(theta), rotation_mode='anchor')
ax.annotate(r"$N=10^{300}$", xy=(3, 1.5*10**4), c=colors[-1], rotation=90-abs(theta), rotation_mode='anchor')
ax2.scatter(logNs, t_less_than, c=colors)
ax2.plot(logNs, logNs, c='k', ls='--')
ax.set_xlim([10**-3, 5*10**3])
ax.set_ylim([1, 10**5])
fig.savefig("Mean.png")

def einstein_var(t, N):
    logN = np.log(N).astype(float)
    return np.piecewise(t,
                        [t <= logN, t > logN],
                        [lambda time: 0, lambda time: np.pi**2 / 6 * (time/logN-1)**2 / (2*time/logN-1)])

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t/\ln N$")
ax.set_ylabel(r"Variance")
N = 300
cdf_df, max_df = db.getMeanVarN(N)
N = np.quad(f"1e{N}")
logN = np.log(N).astype(float)
einstein_theory = einstein_var(cdf_df['time'].values, N)
max_df['Var Max'] = max_df['Var Max'] * 4
w = 50
env_recovered = max_df['Var Max'] - einstein_var(max_df['time'].values, N)
env_recovered = np.convolve(env_recovered, np.ones(w), mode='valid') / w
env_time = np.convolve(max_df['time'].values, np.ones(w), mode='valid') / w
alpha=0.7
theory_var = theory.quantileVar(N, max_df['time'].values, crossover=logN**(3/2), width=logN**(4/3))
ax.plot(max_df['time'] / logN, max_df['Var Max'], c='r', label=r'$Var(Max_t^{(N)})$', alpha=alpha)
ax.plot(cdf_df['time'] / logN, cdf_df['Var Quantile'], c='g', label=r'$Var(Env_t^{(N)})$', alpha=alpha)
ax.plot(cdf_df['time'] / logN, cdf_df['Gumbel Mean Variance'], c='m', label='$Var(Sam_t^{(N)})$', alpha=alpha)
ax.plot(env_time / logN, env_recovered, zorder=2, label=r'$Var(Max_t^{N}) - Var(Sam_t^{(N)})$', c='tab:orange', alpha=alpha)
ax.plot(cdf_df['time'] / logN, einstein_var(cdf_df['time'].values, N), ls='--', c='m')
ax.plot(max_df['time'] / logN, theory_var, 'g--')
ax.plot(max_df['time'] / logN, theory_var + einstein_var(max_df['time'].values, N), ls='--', c='r')
ax.set_xlim([0.5, 10**4])
ax.set_ylim([0.5, 10**4])

ax.set_title(r"$N=10^{300}$")
ax.legend()
fig.savefig("IvanPicture.png")
