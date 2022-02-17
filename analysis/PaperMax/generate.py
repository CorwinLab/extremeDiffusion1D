import sys
sys.path.append("../../src")

from overalldatabase import Database
from matplotlib import pyplot as plt
import theory
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import npquad
import os

def einstein_var(t, N):
    logN = np.log(N).astype(float)
    return np.piecewise(t,
                        [t <= logN, t > logN],
                        [lambda time: 0, lambda time: np.pi**2 / 6 * (time/logN-1)**2 / (2*time/logN-1)])

def getLessThanT(time, mean):
    greater = mean >= time-1
    nonzero = np.nonzero(greater)[0][-1]
    return time[nonzero]

db = Database()
directory = "/home/jacob/Desktop/corwinLabMount/CleanData/Paper/Max/"
cdf_path = "/home/jacob/Desktop/corwinLabMount/CleanData/Paper/CDF/"
dirs = os.listdir(directory)
for dir in dirs:
    path = os.path.join(directory, dir)
    db.add_directory(path, dir_type='Max')
    N = int(path.split('/')[-1])
    #db.calculateMeanVar(path, verbose=True)

db.add_directory(cdf_path, dir_type='Gumbel')
#db.calculateMeanVar(cdf_path, verbose=True, maxTime=3453876)

quantiles = db.N(dir_type='Max')
'''
Make the maximum variance and mean plots
'''
fontsize = 12
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t / \lnN$", labelpad=0, fontsize=fontsize)
ax.set_ylabel(r"$\mathrm{Var}(\mathrm{max}_t^{(N)})$", fontsize=fontsize, labelpad=0)
ax.tick_params(axis='both', labelsize=fontsize)

ax2 = fig.add_axes([0.53, 0.21, 0.35, 0.35])
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel(r"$t / \lnN$", labelpad=0, fontsize=fontsize)
ax2.set_ylabel(r"$\mathrm{Mean}(\mathrm{Max}_t^{(N)})$", labelpad=0, fontsize=fontsize)
ax2.tick_params(axis='both', which='major', labelsize=fontsize)
ax2.set_xlim([10**-3, 5*10**3])
ax2.set_ylim([1, 10**5])

cm = LinearSegmentedColormap.from_list('rg', ['tab:orange', 'tab:red', "tab:purple", 'tab:blue'], N=256)
colors = [
    cm(1.0 * i / len(quantiles) / 1) for i in range(len(quantiles))
]
ypower = 0
for i, N in enumerate(quantiles):
    cdf_df, max_df = db.getMeanVarN(N)

    Nquad = np.quad(f"1e{N}")
    logN = np.log(Nquad).astype(float)
    max_df['Var Max'] = max_df['Var Max'] * 4
    max_df['Mean Max'] = max_df['Mean Max'] * 2

    var_theory = theory.quantileVar(Nquad, max_df['time'].values, crossover=logN**(1.5), width=logN**(4/3))
    mean_theory = theory.quantileMean(Nquad, max_df['time'].values)

    ax.plot(max_df['time'] / logN, (var_theory + theory.gumbel_var(max_df['time'].values, Nquad)) / logN**(ypower), '--', c=colors[i])
    ax.plot(max_df['time'] / logN, max_df['Var Max'] / logN**(ypower), label=N, c=colors[i], alpha=0.5)

    ax2.plot(max_df['time'] / logN, max_df['Mean Max'], c=colors[i], alpha=0.8)
    ax2.plot(max_df['time'] / logN, mean_theory, '--', c=colors[i])

#x, y
start_coord = (20, 6)
end_coord = (6, 9*10**2)
dx = np.log(start_coord[0]) - np.log(end_coord[0])
dy = np.log(start_coord[1]) - np.log(end_coord[1])
theta = np.rad2deg(np.arctan2(dy, dx))+5
ax.annotate("", xy=end_coord, xytext=start_coord,
        arrowprops=dict(shrink=0., facecolor='gray', edgecolor='white', width=50, headwidth=80, headlength=40, alpha=0.3), zorder=0)
ax.annotate(r"$N=10^{2}$", xy=(15, 3), c=colors[0], rotation=90-abs(theta), rotation_mode='anchor')
ax.annotate(r"$N=10^{300}$", xy=(3.5, 8*10**2), c=colors[-1], rotation=90-abs(theta), rotation_mode='anchor')

ax.set_xlim([0.3, 5*10**3])
ax.set_ylim([10**-1, 10**4])
ax2.remove()
fig.savefig("MaxVar.png", bbox_inches='tight')
fig.savefig("./TalkPictures/MaxVar.png", bbox_inches='tight')

'''
Make plot showing the recovery of environmental data
'''
fontsize = 12
alpha=0.7
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t / \lnN$", labelpad=0, fontsize=fontsize)
ax.set_ylabel(r"$\mathrm{Variance}$", fontsize=fontsize)
ax.tick_params(axis='both', labelsize=fontsize)

N = 85
cdf_df, max_df = db.getMeanVarN(N)

Nquad = np.quad(f"1e{N}")
logN = np.log(Nquad).astype(float)

max_df['Var Max'] = max_df['Var Max'] * 4
max_df['Mean Max'] = max_df['Mean Max'] * 2

w = 50
env_recovered = max_df['Var Max'] - einstein_var(max_df['time'].values, Nquad)
env_recovered = np.convolve(env_recovered, np.ones(w), mode='valid') / w
env_time = np.convolve(max_df['time'].values, np.ones(w), mode='valid') / w

var_theory = theory.quantileVar(Nquad, cdf_df['time'].values, crossover=logN**(1.5), width=logN**(4/3))
max_var_theory = theory.quantileVar(Nquad, max_df['time'].values, crossover=logN**(1.5), width=logN**(4/3)) + einstein_var(max_df['time'].values, Nquad)
ax.plot(cdf_df['time'] / logN, cdf_df['Var Quantile'], c='b', alpha=alpha)
ax.plot(cdf_df['time'] / logN, var_theory, ls='--', c='b')
ax.plot(max_df['time'] / logN, max_df['Var Max'], c='r', alpha=alpha)
ax.plot(max_df['time'] / logN, max_var_theory, c='r', ls='--')
ax.plot(cdf_df['time'] / logN, einstein_var(cdf_df['time'].values, Nquad), c='g', ls='--')
ax.plot(cdf_df['time'] / logN, cdf_df['Gumbel Mean Variance'], c='g', alpha=alpha)
ax.plot(env_time / logN, env_recovered, c='tab:orange')
ax.set_xlim([0.3, 5*10**3])
ax.set_ylim([10**-1, 10**4])
fig.savefig("MaxQuantComp.png", bbox_inches='tight')
fig.savefig("./TalkPictures/EnvComp.png", bbox_inches='tight')

'''
Make plot showing quantile variance
'''
fontsize = 12
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t / \lnN$", labelpad=0, fontsize=fontsize)
ax.set_ylabel(r"$\mathrm{Var}(\mathrm{Env}_t^{(N)})$", fontsize=fontsize, labelpad=0)
ax.tick_params(axis='both', labelsize=fontsize)

ax2 = fig.add_axes([0.53, 0.21, 0.35, 0.35])
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel(r"$t / \lnN$", labelpad=0, fontsize=fontsize)
ax2.set_ylabel(r"$\mathrm{Mean}(\mathrm{Env}_t^{(N)})$", labelpad=0, fontsize=fontsize)
ax2.tick_params(axis='both', which='major', labelsize=fontsize)
ax2.set_xlim([10**-3, 5*10**3])
ax2.set_ylim([1, 10**5])

cm = LinearSegmentedColormap.from_list('rg', ['tab:orange', 'tab:red', "tab:purple", 'tab:blue'], N=256)
colors = [
    cm(1.0 * i / len(quantiles) / 1) for i in range(len(quantiles))
]
ypower = 0
for i, N in enumerate(quantiles):
    cdf_df, max_df = db.getMeanVarN(N)
    max_df['Var Max'] = max_df['Var Max'] * 4
    max_df['Mean Max'] = max_df['Mean Max'] * 2

    Nquad = np.quad(f"1e{N}")
    logN = np.log(Nquad).astype(float)

    var_theory = theory.quantileVar(Nquad, cdf_df['time'].values, crossover=logN**(1.5), width=logN**(4/3))
    mean_theory = theory.quantileMean(Nquad, cdf_df['time'].values)

    w = 25
    env_recovered = max_df['Var Max'] - einstein_var(max_df['time'].values, Nquad)
    env_recovered = np.convolve(env_recovered, np.ones(w), mode='valid') / w
    env_time = np.convolve(max_df['time'].values, np.ones(w), mode='valid') / w

    ax.plot(cdf_df['time'] / logN, var_theory / logN**(ypower), '--', c=colors[i])
    ax.plot(cdf_df['time'] / logN, cdf_df['Var Quantile'] / logN**(ypower), label=N, c=colors[i], alpha=0.5)
    ax.plot(env_time / logN, env_recovered, c=colors[i])

    ax2.plot(cdf_df['time'] / logN, cdf_df['Mean Quantile']-2, c=colors[i], alpha=0.8)
    ax2.plot(cdf_df['time'] / logN, mean_theory, '--', c=colors[i])

#x, y
start_coord = (250, 6)
end_coord = (75, 3*10**3)
dx = np.log(start_coord[0]) - np.log(end_coord[0])
dy = np.log(start_coord[1]) - np.log(end_coord[1])
theta = np.rad2deg(np.arctan2(dy, dx))+5
ax.annotate("", xy=end_coord, xytext=start_coord,
        arrowprops=dict(shrink=0., facecolor='gray', edgecolor='white', width=50, headwidth=90, headlength=50, alpha=0.3), zorder=0)
ax.annotate(r"$N=10^{2}$", xy=(190, 3), c=colors[0], rotation=90-abs(theta), rotation_mode='anchor')
ax.annotate(r"$N=10^{300}$", xy=(45, 3*10**3), c=colors[-1], rotation=90-abs(theta), rotation_mode='anchor')

ax.set_xlim([0.3, 5*10**3])
ax.set_ylim([10**-1, 10**4])
ax2.remove()
fig.savefig("QuantileVar.png", bbox_inches='tight')
fig.savefig("./TalkPictures/QuantileVar.png", bbox_inches='tight')

"""
Make a couple of figures for the talk
"""
'''
Short time plot
'''
fontsize = 12
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t / \lnN$", labelpad=0, fontsize=fontsize)
ax.set_ylabel(r"$\mathrm{Var}(\mathrm{Env}_t^{(N)})$", fontsize=fontsize, labelpad=0)
ax.tick_params(axis='both', labelsize=fontsize)

N = 85
i=3
color='r'
cdf_df, _ = db.getMeanVarN(N)

Nquad = np.quad(f"1e{N}")
logN = np.log(Nquad).astype(float)
time = cdf_df['time'].values
var_theory = np.piecewise(time, [time < logN, time >= logN], [lambda t: 0, lambda t: theory.quantileVarShortTime(Nquad, t)])

ax.plot(cdf_df['time'] / logN, var_theory / logN**(ypower), '--', c=color)
ax.plot(cdf_df['time'] / logN, cdf_df['Var Quantile'] / logN**(ypower), label=N, c=colors[i], alpha=0.5)

ax.set_xlim([0.3, 10**4])
ax.set_ylim([10**-1, 10**4])
fig.savefig("./TalkPictures/ShortVar.png", bbox_inches='tight')

'''
Long time plot
'''
fontsize = 12
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t / \lnN$", labelpad=0, fontsize=fontsize)
ax.set_ylabel(r"$\mathrm{Var}(\mathrm{Env}_t^{(N)})$", fontsize=fontsize, labelpad=0)
ax.tick_params(axis='both', labelsize=fontsize)

N = 85
i = 3
cdf_df, _ = db.getMeanVarN(N)

Nquad = np.quad(f"1e{N}")
logN = np.log(Nquad).astype(float)
time = cdf_df['time'].values
var_theory = np.piecewise(time, [time < logN, time >= logN], [lambda t: 0, lambda t: theory.quantileVarShortTime(Nquad, t)])
var_long = np.piecewise(time, [time < logN, time >= logN], [lambda t: np.nan, lambda t: theory.quantileVarLongTime(Nquad, t)])

ax.plot(cdf_df['time'] / logN, var_theory / logN**(ypower), '--', c='r')
ax.plot(cdf_df['time'] / logN, var_long / logN **ypower, '--', c='b')
ax.plot(cdf_df['time'] / logN, cdf_df['Var Quantile'] / logN**(ypower), label=N, c=colors[i], alpha=0.5)

ax.set_xlim([0.3, 10**4])
ax.set_ylim([10**-1, 10**4])
fig.savefig("./TalkPictures/LongVar.png", bbox_inches='tight')

'''
Fit time plot
'''
fontsize = 12
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t / \lnN$", labelpad=0, fontsize=fontsize)
ax.set_ylabel(r"$\mathrm{Var}(\mathrm{Env}_t^{(N)})$", fontsize=fontsize, labelpad=0)
ax.tick_params(axis='both', labelsize=fontsize)

N = 85
i=3
cdf_df, _ = db.getMeanVarN(N)
ypower = 0

Nquad = np.quad(f"1e{N}")
logN = np.log(Nquad).astype(float)
time = cdf_df['time'].values
var_theory = theory.quantileVar(Nquad, cdf_df['time'].values, crossover=logN**(1.5), width=logN**(4/3))

ax.plot(cdf_df['time'] / logN, var_theory / logN**(ypower), '--', c=colors[i])
ax.plot(cdf_df['time'] / logN, cdf_df['Var Quantile'] / logN**(ypower), label=N, c=colors[i], alpha=0.5)

ax.set_xlim([0.3, 10**4])
ax.set_ylim([10**-1, 10**4])
fig.savefig("./TalkPictures/ShortLongVar.png", bbox_inches='tight')
