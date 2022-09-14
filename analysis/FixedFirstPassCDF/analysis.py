import numpy as np
import npquad
from matplotlib import pyplot as plt
import glob 
import pandas as pd
import os
import sys 
sys.path.append("../../dataAnalysis")
from theory import KPZ_var_fit

def prefactor(x, N):
    logN = np.log(N)
    return 1/2 * (1 + logN / x)

def var_theory(x, N):
    logN = np.log(N).astype(float)
    return x**4 / 4 / logN**4 * KPZ_var_fit(8 * logN**3 / x**2) #* prefactor(x, N)

def I(v): 
    return 1-np.sqrt(1-v**2)

def sigma(v): 
    return (2 * I(v)**2 / (1-I(v)))**(1/3)

def var_short(x, N):
    logN = np.log(N).astype(float)
    t0 = (logN**2 + x**2)/ (2*logN)
    return (t0**(1/3) * sigma(x/t0) / (I(x/t0) - x**2 / t0**2 / np.sqrt(1-(x/t0)**2)))**2 * 0.8133 #* prefactor(x, N)
    
def calculateMeanVar(files, max_dist, verbose=True):
    average_data = None
    average_data_squared = None
    number_of_files = 0
    for f in files:
        data = pd.read_csv(f, delimiter=',') # columns are position, quantile, variance
        
        if max(data['Position']) < max_dist:
            continue
        '''The N=1e2 data got some differet values for position so this is a quick hack
        if data.shape != (356, 3):
            continue
        '''
        data = data.values
        number_of_files += 1
        if average_data is None:
            average_data = data
            average_data_squared = data ** 2
        else:
            average_data += data
            average_data_squared += data ** 2
        print(f)
    print(number_of_files)
    mean = average_data / number_of_files
    variance = average_data_squared / number_of_files - mean ** 2
    return data[:, 0], mean, variance 
    

dir = "/home/jacob/Desktop/talapasMount/JacobData/FixedFirstPassCDF/24/F*.txt"
files = glob.glob(dir)
run_again = False

if run_again:
    position, mean, variance = calculateMeanVar(files, 55262)
    env_variance = variance[:, 1]
    sam_variance = mean[:, 2]
    env_mean24 = mean[:, 1]
    np.savetxt("Position.txt", position)
    np.savetxt("Environmental.txt", env_variance)
    np.savetxt("SamplingVariance.txt", sam_variance)
    np.savetxt("EnvironmentalMean24.txt", env_mean24)
else: 
    position = np.loadtxt("Position.txt")
    env_variance = np.loadtxt("Environmental.txt")
    sam_variance = np.loadtxt("SamplingVariance.txt")
    env_mean24 = np.loadtxt("EnvironmentalMean24.txt")


dir = "/home/jacob/Desktop/talapasMount/JacobData/FixedFirstPassCDF/7/F*.txt"
files = glob.glob(dir)
run_again = False

if run_again:
    position7, mean, variance = calculateMeanVar(files, 16118)
    env_variance7 = variance[:, 1]
    sam_variance7 = mean[:, 2]
    env_mean7 = mean[:, 1]
    np.savetxt("Position7.txt", position7)
    np.savetxt("Environmental7.txt", env_variance7)
    np.savetxt("EnvironmentalMean7.txt", env_mean7)
    np.savetxt("SamplingVariance7.txt", sam_variance7)
else: 
    position7 = np.loadtxt("Position7.txt")
    env_variance7 = np.loadtxt("Environmental7.txt")
    sam_variance7 = np.loadtxt("SamplingVariance7.txt")
    env_mean7 = np.loadtxt("EnvironmentalMean7.txt")

distanceMax = np.loadtxt("../FirstPassDiscreteAbs/distances.txt")
varianceMax = np.loadtxt("../FirstPassDiscreteAbs/varianceMax.txt")
good_idx = (varianceMax > 10**-3)  * (varianceMax < varianceMax[-1])
varianceMax = varianceMax[good_idx]
distanceMax = distanceMax[good_idx]

N = np.quad("1e24")
logN = np.log(N).astype(float)

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim([0.5, 500])
ax.set_ylim([10**-3, 10**11])
ax.set_xlabel(r"$x / \log(N)$")
ax.set_ylabel(r"$\mathrm{Var}(\tau)$")
ax.plot(position / logN, env_variance, c='r')
ax.plot(position / logN, sam_variance, c='b')
ax.plot(distanceMax / logN, varianceMax, c='k', ls='--', alpha=0.5)
ax.plot(position / logN, env_variance + sam_variance)

fig.savefig("Variance.pdf", bbox_inches='tight')

dir = "/home/jacob/Desktop/talapasMount/JacobData/FixedFirstPassCDF/2/F*.txt"
files = glob.glob(dir)
run_again = False

if run_again:
    position2, mean, variance = calculateMeanVar(files, 4605)
    env_variance2 = variance[:, 1]
    sam_variance2 = mean[:, 2]
    env_mean2 = mean[:, 1]
    np.savetxt("Position2.txt", position2)
    np.savetxt("Environmental2.txt", env_variance2)
    np.savetxt("SamplingVariance2.txt", sam_variance2)
    np.savetxt("EnvironmentalMean2.txt", env_mean2)
else: 
    position2 = np.loadtxt("Position2.txt")
    env_variance2 = np.loadtxt("Environmental2.txt")
    sam_variance2 = np.loadtxt("SamplingVariance2.txt")
    env_mean2 = np.loadtxt("EnvironmentalMean2.txt")

Nquant_data = pd.read_csv("../LocustFirstPassTest/Total_Times.csv")
unique_distances = np.unique(Nquant_data['Distances'])
vars = []
means = []
for d in unique_distances:
    data = Nquant_data[Nquant_data['Distances'] == d]
    times = data['Time'].values
    vars.append(np.var(times))
    means.append(np.mean(times))

vars_first = []
means_first = []
for d in unique_distances:
    data = Nquant_data[Nquant_data['Distances'] == d]
    data = data[data['Number Crossed'] == 0]
    times = data['Time'].values 
    vars_first.append(np.var(times))
    means_first.append(np.mean(times))

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$x/ \log(N)$")
ax.set_xlim([0.5, 1000])
ax.set_ylabel(r"$\frac{\mathrm{Var}(\tau_{\mathrm{Env}})}{\sqrt{\log(N)}}$")
ax.plot(position2 / np.log(1e2), env_variance2/(np.sqrt(np.log(1e2))), c='b', label=r'$N=10^2$', alpha=0.5)
ax.plot(position7 / np.log(1e7), env_variance7 / np.sqrt(np.log(1e7)), c='m', label=r'$N=10^7$', alpha=0.5)
ax.plot(position / np.log(1e24), env_variance/(np.sqrt(np.log(1e24))), c='r', label=r'$N=10^{24}$', alpha=0.5)

theory_pos = position[position > np.log(1e24)]
short = var_short(theory_pos, 1e24)
jacob_var = var_theory(theory_pos, 1e24)
ax.plot(theory_pos / np.log(1e24), jacob_var / (np.sqrt(np.log(1e24))), '--', label=r'KPZ Regime $N=10^{24}$')
ax.plot(theory_pos / np.log(1e24), short / (np.sqrt(np.log(1e24))), '--', label=r'TW Regime $N=10^{24}$')
ax.plot(theory_pos / np.log(1e24), theory_pos ** (8/3) / np.log(1e24) ** (2) * 0.813 * 2**(-2/3) / (np.sqrt(np.log(1e24))), '-.', label=r'$\frac{x^{8/3}}{2^{2/3}\log(N)^{2}}\mathrm{Var}(\chi)$')
ax.plot(unique_distances / np.log(1e7), vars / np.sqrt(np.log(1e7)), label=r'$10^7$ Quantile', alpha=0.6)
ax.plot(unique_distances / np.log(1e7), vars_first / np.sqrt(np.log(1e7)), label=r'First $10^7$ Quantile', alpha=0.6)

ax.legend()
fig.savefig("EnvVariance.pdf", bbox_inches='tight')

fig, ax = plt.subplots()
ax.plot(position2 / np.log(1e2), env_mean2, c='b', label=r'$N=10^2$')
ax.plot(position7 / np.log(1e7), env_mean7, c='m', label=r'$N=10^7$')
ax.plot(position / np.log(1e24), env_mean24, c='r', label=r'$N=10^{24}$')

ax.plot(unique_distances / np.log(1e7), means, '--', label=r'$10^7$ Quantile')
ax.plot(unique_distances / np.log(1e7), means_first, '-.', label=r'First $10^7$ Quantile')

ax.set_xlim([10**-2, 10**3])
ax.set_ylim([1, 5*10**7])
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$x / log(N)$")
ax.set_ylabel(r"$Mean(\tau_{Env})$")
ax.legend()
fig.savefig("EnvMean.pdf", bbox_inches='tight')