import numpy as np
import os 
import sys
sys.path.append("../../dataAnalysis")
from fptTheory import variance, sam_variance_theory
from matplotlib import pyplot as plt
import glob
import pandas as pd

def calculateMeanVarDiscrete(files, max_dist, verbose=True):
    average_data = None
    average_data_squared = None
    number_of_files = 0
    pos = None
    for f in files:
        try:
            data = np.loadtxt(f, delimiter=',', skiprows=1) # columns are distance, time
        except:
            print("Error with:", f)
            continue
        if max(data[:, 0]) < max_dist:
            continue
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
        data = np.loadtxt(f, delimiter=',', skiprows=1) # columns are Position, Quantile, Variance
        if max(data[:, 0]) < max_dist:
            continue
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
    new_df = pd.DataFrame(np.array([pos, mean[:, 1], mean[:, 2], variance[:, 1]]).T, columns=['Distance', 'Mean Quantile', 'Sampling Variance', 'Env Variance'])
    return new_df, number_of_files

home_dir = '/home/jacob/Desktop/talapasMount/JacobData/FPTDiscretePaper'
dirs = os.listdir(home_dir)
max_dists = [2301, 4605, 11512, 27631]
Ns = [1, 2, 5, 12] #, 28]
recalculate_mean = False
if recalculate_mean:
    nFiles = []
    for max_dist, N in zip(max_dists, Ns):
        dir = home_dir + f'/{N}/Q*.txt'
        files = glob.glob(dir)
        df, number_of_files = calculateMeanVarDiscrete(files, max_dist)
        path = os.path.join(home_dir,f'{N}', 'MeanVariance.csv')
        df.to_csv(path, index=False)
        nFiles.append(number_of_files)

    np.savetxt("NumberOfSystems.txt", nFiles)

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
    np.savetxt(os.path.join(cdf_dir, f'{N}', 'NumberOfSystems.txt'), nFiles)

colors = ['b', 'g', 'r', 'm']
alpha = 0.6
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$Var(\tau_{Max})$")
ax.set_xlim([1, max(max_dists)])
ax.set_ylim([10**-2, 10**11 * 2])
for i, Nexp in enumerate(Ns): 
    dir = home_dir + f'/{Nexp}/'
    meanFile = os.path.join(dir, 'MeanVariance.csv')
    df = pd.read_csv(meanFile)
    N = float(f'1e{Nexp}')
    logN = np.log(N)

    var_theory = variance(df['Distance'].values, N) + sam_variance_theory(df['Distance'].values, N)

    ax.plot(df['Distance'], df['Variance'], c=colors[i], alpha=alpha, label=f'N={N}')
    ax.plot(df['Distance'], var_theory, ls='--', c=colors[i])
ax.legend()
fig.savefig("MaxVariance.pdf", bbox_inches='tight')

colors = ['b', 'g', 'r', 'm']
alpha = 0.6
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$Mean(\tau_{Max})$")
ax.set_xlim([1/np.log(1e12), 10**3])
ax.set_ylim([1, 3*10**7])
for i, Nexp in enumerate(Ns): 
    dir = home_dir + f'/{Nexp}/'
    meanFile = os.path.join(dir, 'MeanVariance.csv')
    df = pd.read_csv(meanFile)
    N = float(f'1e{Nexp}')
    logN = np.log(N)
    var_theory = variance(df['Distance'].values, N) + sam_variance_theory(df['Distance'].values, N)
    ax.plot(df['Distance'] / logN, df['Mean'], c=colors[i], alpha=alpha, label=f'N={N}')
    #ax.plot(df['Distance'], var_theory, ls='--', c=colors[i])
ax.legend()
fig.savefig("MaxMean.pdf", bbox_inches='tight')

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$x / \log(N)$")
ax.set_ylabel(r"$\mathrm{Variance}$")
ax.set_xlim([0.5, 10**3])

max_color = "tab:red"
quantile_color = "tab:blue"
gumbel_color = "tab:green"
einsten_color = "tab:purple"
alpha = 0.75

Nexp = 12
N = float(f'1e{Nexp}')
logN = np.log(N)

max_dir = home_dir + f'/{Nexp}'
max_file = os.path.join(max_dir, 'MeanVariance.csv')
max_df = pd.read_csv(meanFile)

cdf_dir = cdf_dir + f'/{Nexp}'
cdf_file = os.path.join(cdf_dir, 'MeanVariance.csv')
cdf_df = pd.read_csv(cdf_file)

env_theory = variance(cdf_df['Distance'].values, N)
sam_theory = sam_variance_theory(cdf_df['Distance'].values, N)

ax.plot(max_df['Distance'] / logN, max_df['Variance'], label=r'$\mathrm{Var}(\tau_{\mathrm{Min}})$', c=max_color, alpha=alpha)
ax.plot(cdf_df['Distance'] / logN, cdf_df['Sampling Variance'], label=r'$\mathrm{Var}(\tau_{\mathrm{Sam}})$', c=gumbel_color, alpha=alpha)
ax.plot(cdf_df['Distance'] / logN, cdf_df['Env Variance'], label=r'$\mathrm{Var}(\tau_{\mathrm{Env}})$', c=quantile_color, alpha=alpha)

ax.plot(cdf_df['Distance'] / logN, env_theory, ls='--', c=quantile_color)
ax.plot(cdf_df['Distance'] / logN, sam_theory, ls='--', c=gumbel_color)
ax.plot(cdf_df['Distance'] / logN, env_theory + sam_theory, ls='--', c=max_color)

leg = ax.legend(
    loc="upper left",
    framealpha=0,
    labelcolor=[max_color, gumbel_color, quantile_color],
    handlelength=0,
    handletextpad=0,
)

for item in leg.legendHandles:
    item.set_visible(False)
fig.savefig("CompleteVariance.pdf", bbox_inches='tight')