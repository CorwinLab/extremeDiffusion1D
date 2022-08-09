import sys
sys.path.append("../../src")

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import glob
import os
import pandas as pd
from scipy.stats import linregress

def calculate_mean(files, save_file, verbose=True):
    df_tot = pd.DataFrame()
    for f in files:
        df = pd.read_csv(f, delimiter=',')
        if max(df['Distance']) != 5526:
            continue

        df_tot = pd.concat([df_tot, df])
        if verbose:
            print(f)

    df_tot.reset_index(inplace=True, drop=True)
    df_tot.to_csv(save_file)
    return df_tot

def calculate_distance_mean(df, mean_save, var_save):
    unique_distances = np.unique(df['Distance'].values)

    mean = []
    var = []

    for d in unique_distances:
        df_d = df[df['Distance'] == d]
        mean.append(np.mean(df_d['Time'].values))
        var.append(np.var(df_d['Time'].values))

    np.savetxt(mean_save, np.array([unique_distances, mean]))
    np.savetxt(var_save, np.array([unique_distances, var]))
    return unique_distances, mean, var


if __name__ == '__main__':
    dir = "/home/jacob/Desktop/talapasMount/JacobData/FirstPassDiscreteAbs"

    run_again = False
    if run_again:
        path = os.path.join(dir, 'Q*.txt')
        files = glob.glob(path)
        df = calculate_mean(files, os.path.join(dir, "ConcatData.txt"), verbose=True)
        
        mean_file = os.path.join(dir, "Mean.txt")
        var_file = os.path.join(dir, "Var.txt")
        d, m, v = calculate_distance_mean(df, mean_file, var_file)

    fontsize=12
    cm = LinearSegmentedColormap.from_list('rg', ['tab:orange', 'tab:red', "tab:purple", 'tab:blue'], N=256)
    alpha=0.7

    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"$\mathrm{Var}(\tau)$", fontsize=fontsize)
    ax.set_xlabel(r"$x / \log(N)$", fontsize=fontsize)
    min_color ='tab:red'
    sam_color = 'tab:green'
    env_color = 'tab:blue'
    theory_color = 'tab:purple'

    N_exp = 24
    N = float(f"1e{N_exp}")
    logN = np.log(N).astype(float)
    mean_file = os.path.join(dir, "Mean.txt")
    var_file = os.path.join(dir, "Var.txt")
    mean = np.loadtxt(mean_file).T
    var = np.loadtxt(var_file).T
    distance = mean[:, 0]
    assert (var[:, 0] == distance).all()
    var = var[:, 1]
    mean = mean[:, 1]
    ax.plot(distance / logN, var, label=r'$\mathrm{Var}(\tau_{min})$', alpha=alpha, c=min_color)

    RWRESam = np.loadtxt("../FirstPassageCDFLonger/AveragedData.txt")
    theoretical_distances = np.loadtxt("distances.txt")
    theoretical_variance = np.loadtxt("varianceShortTime.txt")
    theoretical_variance_2 = np.loadtxt("varianceLongTime.txt")

    # put this in mathematica code so don't need to do this anymore
    #theoretical_variance_2 = (theoretical_variance_2 / 2)**2
    #theoretical_variance = (theoretical_variance / 2)**2

    theoretical_variance = np.delete(theoretical_variance, np.argmax(theoretical_variance))
    theoretical_distances = np.delete(theoretical_distances, np.argmax(theoretical_variance))
    theoretical_variance_2 = np.delete(theoretical_variance_2, np.argmax(theoretical_variance))
    '''
    res = linregress(np.log(RWRESam[:, 0])[-500:], np.log(RWRESam[:, 4])[-500:])
    slope = res.slope
    intercept = res.intercept
    theoretical_var = np.exp(intercept) * (RWRESam[:,0][-500:] ** slope)
    print(f"{np.exp(intercept)} * x^{slope}")
    '''

    theoretical_distances_max = np.loadtxt("distances.txt")
    theoretical_variance_max = np.loadtxt("varianceMax.txt")

    #ax.plot(RWRESam[:, 0][-500:] / logN, theoretical_var, label=r'$x^{3.8}$', ls='--')
    ax.plot(RWRESam[:, 0] / logN, RWRESam[:, 2], label=r'$\mathrm{Var}(\tau_{sam})$', alpha=alpha, c=sam_color)
    ax.plot(RWRESam[:, 0] / logN, RWRESam[:, 4], label=r'$\mathrm{Var}(\tau_{env})$', alpha=alpha, c=env_color)
    #ax.plot(RWRESam[:, 0] / logN, RWRESam[:, 4] + RWRESam[:, 2], label=r'$\mathrm{Var}(\mathrm{Env}_x^N) + \mathrm{Var}(\mathrm{Sam}_x^N)$', alpha=alpha)
    ax.plot(theoretical_distances / logN, theoretical_variance, label=r'"Theoretical" $\mathrm{Var}(\tau_{env})$', ls='--', c=theory_color)
    ax.plot(theoretical_distances / logN, theoretical_variance_2, label=r'"Theoretical KPZ" $\mathrm{Var}(\tau_{env})$', ls='--', c='saddlebrown')
    ax.plot(theoretical_distances_max / logN, theoretical_variance_max, label=r'"Theoretical Max"', ls='-.', c='k')
    xvals = np.array([50, 90])
    xvals2 = np.array([5, 20])
    yvals = xvals ** 4
    ax.vlines(logN**(3/2)/logN, 10**-2, 10**9, ls='--', color='k', label=r'$(\log(N))^{3/2}$')
    ax.set_ylim([10**-2, 10**9])
    #ax.plot(xvals, yvals / 5, c='k', ls='--', label=r'$x^4$')
    #ax.plot(xvals2, xvals2**(8/3) * 9, c='k', ls='-.', label=r'$x^{8/3}$')
    ax.set_xlim([0.6, max(RWRESam[:,0])/logN])
    leg = ax.legend(fontsize=fontsize, loc='upper left', framealpha=0, labelcolor=[min_color, sam_color, env_color, theory_color, 'saddlebrown', 'k', 'k'], handlelength=0, handletextpad=0)
    fig.savefig("Var.pdf", bbox_inches='tight')
    

