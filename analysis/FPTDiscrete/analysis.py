import numpy as np
import os 
import sys
sys.path.append("../../dataAnalysis")
from fptTheory import variance, sam_variance_theory, var_power_long, var_short, mean_theory
from matplotlib import pyplot as plt
import glob
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams.update({"font.size": 15})

def calculateMeanVarDiscrete(files, max_dist, verbose=True):
    average_data = None
    average_data_squared = None
    number_of_files = 0
    pos = None
    for f in files:
        try:
            data = pd.read_csv(f) # columns are distance, time
        except:
            print("Error with:", f)
            continue
        if max(data['Distance']) < max_dist:
            continue
        data = data[data['Distance'] <= max_dist]
        data = data.values
        pos = data[:, 0]
        number_of_files += 1
        if average_data is None:
            average_data = data
            average_data_squared = data ** 2
        else:
            average_data += data
            average_data_squared += data ** 2
        if verbose:
            print(f)

    if number_of_files == 0:
        return None
    mean = average_data / number_of_files
    variance = average_data_squared / number_of_files - mean ** 2
    new_df = pd.DataFrame(np.array([pos, mean[:, 1], variance[:, 1]]).T, columns=['Distance', 'Mean', 'Variance'])
    return new_df, number_of_files

def calculateMeanVarCDF(files, max_dist, verbose=True):
    average_data = None
    average_data_squared = None
    number_of_files = 0
    pos = None
    for f in files: 
        data = pd.read_csv(f) # columns are Position, Quantile, Variance
        if max(data['Position']) < max_dist:
            print("Not Enough Data: ", f)
            continue
        data = data[data['Position'] <= max_dist]
        data = data.values
        pos = data[:, 0]
        number_of_files += 1
        if average_data is None:
            average_data = data
            average_data_squared = data ** 2
        else:
            average_data += data
            average_data_squared += data ** 2
    if number_of_files == 0:
        return None
    mean = average_data / number_of_files
    variance = average_data_squared / number_of_files - mean ** 2
    new_df = pd.DataFrame(np.array([pos, mean[:, 1], mean[:, 2], variance[:, 1]]).T, columns=['Distance', 'Mean Quantile', 'Sampling Variance', 'Env Variance'])
    return new_df, number_of_files

home_dir = '/home/jacob/Desktop/talapasMount/JacobData/FPTDiscretePaper'
dirs = os.listdir(home_dir)
max_dists = [2301, 4605, 11512, 27631]
Ns = [1, 2, 5, 12, 28]
N_vals = [float(f"1e{i}") for i in Ns]
max_dists = np.array(np.log(N_vals)) * 750
Nlabels = [r'$N=10$', r'$N=10^2$', r'$N=10^{5}$', r'$N=10^{12}$', r'$N=10^{28}$']
recalculate_mean = False
if recalculate_mean:
    nFiles = []
    for max_dist, N in zip(max_dists, Ns):
        dir = home_dir + f'/{N}/Q*.txt'
        files = glob.glob(dir)
        df, number_of_files = calculateMeanVarDiscrete(files, max_dist, verbose=False)
        path = os.path.join(home_dir,f'{N}', 'MeanVariance.csv')
        df.to_csv(path, index=False)
        nFiles.append(number_of_files)
        print(f"Max {N}: {number_of_files} files")

        np.savetxt(home_dir + f'/{N}/NumberOfSystems.txt', [number_of_files])

for N in Ns: 
    num_files = np.loadtxt(home_dir + f'/{N}/NumberOfSystems.txt')
    print(f"Discrete {N}: {num_files} files")

cdf_dir = '/home/jacob/Desktop/corwinLabMount/CleanData/FPTCDFPaper'
dirs = os.listdir(cdf_dir)
recalculate_mean = False
if recalculate_mean: 
    nFiles = []
    for max_dist, N in zip(max_dists, Ns):
        dir = cdf_dir + f'/{N}/First*.txt'
        files = glob.glob(dir)
        df, number_of_files = calculateMeanVarCDF(files, max_dist)
        path = os.path.join(cdf_dir,f'{N}', 'MeanVariance.csv')
        df.to_csv(path, index=False)
        nFiles.append(number_of_files)
        print(f"CDF {N}: {number_of_files} files")

        np.savetxt(os.path.join(cdf_dir, f'{N}', 'NumberOfSystems.txt'), [number_of_files])

for N in Ns: 
    num_files = np.loadtxt(os.path.join(cdf_dir, f'{N}', 'NumberOfSystems.txt'))
    print(f"CDF {N}: {num_files} files")

einstein_files = glob.glob('/home/jacob/Desktop/talapasMount/JacobData/FPTDiscreteE/12/Q*.txt')
max_dist = 750 * np.log(1e12)
recalculate_mean = False
if recalculate_mean: 
    df, number_of_files = calculateMeanVarDiscrete(einstein_files, max_dist)
    df.to_csv('/home/jacob/Desktop/talapasMount/JacobData/MeanVariance.csv', index=False)
    np.savetxt('/home/jacob/Desktop/talapasMount/JacobData/NumberOfSystems.csv', [number_of_files])
    print(number_of_files)
einstein_df = pd.read_csv('/home/jacob/Desktop/talapasMount/JacobData/MeanVariance.csv')
num_files = np.loadtxt('/home/jacob/Desktop/talapasMount/JacobData/NumberOfSystems.csv')
print(f"Einstein Discrete: {num_files} files")

cm = LinearSegmentedColormap.from_list(
    "rg", ["tab:orange", "tab:red", "tab:purple", "tab:blue"], N=256
)
colors = [cm(1.0 * i / len(Ns) / 1) for i in range(len(Ns))]

alpha = 0.6
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$L$")
ax.set_ylabel(r"$\mathrm{Var}(\tau_{\mathrm{Min}})$")
ax.set_xlim([1, max(max_dists)])
ax.set_ylim([10**-2, 10**11])
for i, Nexp in enumerate(Ns): 
    dir = home_dir + f'/{Nexp}/'
    meanFile = os.path.join(dir, 'MeanVariance.csv')
    df = pd.read_csv(meanFile)
    N = float(f'1e{Nexp}')
    logN = np.log(N)

    var_theory = variance(df['Distance'].values, N) + sam_variance_theory(df['Distance'].values, N)

    ax.plot(df['Distance'], df['Variance'], c=colors[i], alpha=alpha, label=Nlabels[i])
    ax.plot(df['Distance'], var_theory, ls='--', c=colors[i])
    #ax.plot(df['Distance'], np.pi**2/24*df['Distance']**4 / logN**4, c=colors[i], ls='-.')

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

alpha = 0.6
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$L / \log(N)$")
ax.set_ylabel(r"$\mathrm{Mean}(\tau_{\mathrm{Min}})$")
ax.set_xlim([1/np.log(1e28), 750])
ax.set_ylim([1, 3*10**7])
for i, Nexp in enumerate(Ns): 
    dir = home_dir + f'/{Nexp}/'
    meanFile = os.path.join(dir, 'MeanVariance.csv')
    df = pd.read_csv(meanFile)
    N = float(f'1e{Nexp}')
    logN = np.log(N)
    mean = mean_theory(df['Distance'].values, N)
    ax.plot(df['Distance'] / logN, df['Mean'], c=colors[i], alpha=alpha, label=Nlabels[i])
    ax.plot(df['Distance'] / logN, mean, c=colors[i], ls='--')

leg = ax.legend(
    loc="upper left",
    framealpha=0,
    labelcolor=colors,
    handlelength=0,
    handletextpad=0,
)
for item in leg.legendHandles:
    item.set_visible(False)

fig.savefig("MaxMean.pdf", bbox_inches='tight')

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$L / \log(N)$")
ax.set_ylabel(r"$\mathrm{Variance}$")
ax.set_xlim([0.5, 750])

max_color = "tab:red"
quantile_color = "tab:blue"
gumbel_color = "tab:green"
einstein_color = "tab:purple"
alpha = 0.75

Nexp = 12
dir = home_dir + f'/{Nexp}/'
meanFile = os.path.join(dir, 'MeanVariance.csv')
max_df = pd.read_csv(meanFile)
N = float(f'1e{Nexp}')
logN = np.log(N)

cdf_dir_12 = cdf_dir + f'/{Nexp}'
cdf_file = os.path.join(cdf_dir_12, 'MeanVariance.csv')
cdf_df = pd.read_csv(cdf_file)

env_theory = variance(cdf_df['Distance'].values, N)
sam_theory = sam_variance_theory(cdf_df['Distance'].values, N)
new_row = [np.log2(1e12), 0]
einstein_theoretical_data = np.loadtxt("etheoretical12.txt")
einstein_theoretical_data = np.vstack([new_row, einstein_theoretical_data])

ax.plot(max_df['Distance'] / logN, max_df['Variance'], label=r'$\mathrm{Var}(\tau_{\mathrm{Min}})$', c=max_color, alpha=alpha)
ax.plot(cdf_df['Distance'] / logN, cdf_df['Sampling Variance'], label=r'$\mathrm{Var}(\tau_{\mathrm{Sam}})$', c=gumbel_color, alpha=alpha)
ax.plot(cdf_df['Distance'] / logN, cdf_df['Env Variance'], label=r'$\mathrm{Var}(\tau_{\mathrm{Env}})$', c=quantile_color, alpha=alpha)
ax.plot(einstein_df['Distance'] / logN, einstein_df['Variance'], label=r'$\mathrm{Var}(\tau_{\mathrm{SSRW}})$', c=einstein_color, alpha=alpha)

ax.plot(einstein_theoretical_data[:, 0] / logN, einstein_theoretical_data[:, 1], ls='--', c=einstein_color)
ax.plot(cdf_df['Distance'] / logN, env_theory, ls='--', c=quantile_color)
ax.plot(cdf_df['Distance'] / logN, sam_theory, ls='--', c=gumbel_color)
ax.plot(cdf_df['Distance'] / logN, env_theory + sam_theory, ls='--', c=max_color)

xvals = np.array([100, 600])
ax.plot(xvals, xvals ** 4, label=r'$L^{4}$', c='k', ls='--')
ax.plot(xvals, xvals ** 3, label=r'$L^{3}$', c='k', ls='--')

leg = ax.legend(
    loc="upper left",
    framealpha=0,
    labelcolor=[max_color, gumbel_color, quantile_color, einstein_color, 'k', 'k'],
    handlelength=0,
    handletextpad=0,
)
ax.set_ylim([10**-3, 10**12])
for item in leg.legendHandles:
    item.set_visible(False)
fig.savefig("CompleteVariance.pdf", bbox_inches='tight')

# Plotting the SSRW data
theoretical_data = np.loadtxt("Data.txt")
N = float(10**24)
logN = np.log2(N)
theoretical_curve = sam_variance_theory(theoretical_data[:, 0], N)
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$x / \log_{2}(N)$")
ax.set_ylabel(r"$\mathrm{Variance}$")
ax.plot(theoretical_data[:, 0] / logN, theoretical_data[:, 1], label=r'$\mathrm{Var}(\tau_{\mathrm{SSRW}})$', alpha=0.75)
ax.plot(theoretical_data[:, 0] / logN, theoretical_curve, label=r'$\mathrm{Var}(\tau_{\mathrm{Sam}})$', ls='--', alpha=0.75)
ax.legend()
fig.savefig("SSRW.pdf", bbox_inches='tight')

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$L / \log(N)$")
ax.set_ylabel(r"$\mathrm{Var}(\tau_{\mathrm{Env}})$")
ax.set_xlim([0.5, 10**3])
ax.set_ylim([10**-2, 10**10])

for i, Nexp in enumerate(Ns):
    cdf_dir_specific = cdf_dir + f'/{Nexp}'
    N = float(f"1e{Nexp}")
    logN = np.log(N)
    cdf_file = os.path.join(cdf_dir_specific, 'MeanVariance.csv')
    cdf_df = pd.read_csv(cdf_file)
    var_theory = variance(cdf_df['Distance'].values, N)
    ax.plot(cdf_df['Distance'] / logN, var_theory, c=colors[i], ls='--')
    ax.plot(cdf_df['Distance'] / logN, cdf_df['Env Variance'], label=Nlabels[i], c=colors[i], alpha=alpha)

xvals = np.array([100, 600])
ax.plot(xvals, 10 * xvals ** 3, c='k', ls='--', label=r'$L^3$')

leg = ax.legend(
    loc="upper left",
    framealpha=0,
    labelcolor=colors + ['k'],
    handlelength=0,
    handletextpad=0,
)
ax.set_ylim([10**-3, 10**12])
for item in leg.legendHandles:
    item.set_visible(False)

fig.savefig("EnvironmentalVariance.pdf", bbox_inches='tight')


alpha = 0.6
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$L / \log(N)$")
ax.set_ylabel(r"$\mathrm{Var}(\tau_{\mathrm{Env}})$")
ax.set_xlim([0.5, 10**3])
ax.set_ylim([10**-1, 10**10])
Nexp = 12
cdf_dir_specific = cdf_dir + f'/{Nexp}'
N = float(f"1e{Nexp}")
logN = np.log(N)
cdf_file = os.path.join(cdf_dir_specific, 'MeanVariance.csv')
cdf_df = pd.read_csv(cdf_file)
ax.plot(cdf_df['Distance'] / logN, cdf_df['Env Variance'], c=colors[-2], alpha=alpha)
theory_distances = cdf_df[cdf_df['Distance'] >= logN]['Distance'].values
long = var_power_long(theory_distances, N)
short = var_short(theory_distances, N)
var = variance(theory_distances, N)
ax.plot(theory_distances / logN, short, ls='--', c='tab:red', label=r'$V_1(L, N)$')
ax.plot(theory_distances / logN, long, ls='--', c='tab:green', label=r'$V_2(L, N)$')
ax.plot(theory_distances / logN, var, ls='--', c='k', zorder=-1)
ax.vlines(logN**(3/2) / logN, 10**-1, 10**10, ls='--', color='k')
ax.annotate(r'$L=\log(N)^{3/2}$', (3.5, 10**6), rotation=90)

leg = ax.legend(
    loc="lower right",
    framealpha=0,
    labelcolor=['tab:red', 'tab:green'],
    handlelength=0,
    handletextpad=0,
)
for item in leg.legendHandles:
    item.set_visible(False)

'''This isn't helpful
axins = ax.inset_axes([0.53, 0.05, 0.45, 0.45])
axins.plot(theory_distances / logN, short, ls='--', c='tab:red')
axins.plot(theory_distances / logN, long, ls='--', c='tab:green')
axins.plot(theory_distances / logN, var, ls='--', c='k', zorder=-1)
axins.plot(cdf_df['Distance'] / logN, cdf_df['Env Variance'], c=colors[-2], alpha=alpha)
mult=2
axins.set_xlim([5, 6])
axins.set_ylim([2*10 ** 2, 3 * 10 ** 2])
axins.set_xscale("log")
axins.set_yscale("log")
axins.xaxis.set_ticklabels([])
axins.yaxis.set_ticklabels([])
ax.indicate_inset_zoom(axins, edgecolor="black")
'''
fig.savefig("EnvironmentalStitching.pdf", bbox_inches='tight')