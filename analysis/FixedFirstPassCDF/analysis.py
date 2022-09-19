import numpy as np
import npquad
from matplotlib import pyplot as plt
import glob 
import pandas as pd
from TracyWidom import TracyWidom
from scipy.special import erf
import os

def I(v):
    return 1 - np.sqrt(1-v**2)

def Iprime(v):
    return v / np.sqrt(1-v**2)

def sigma(v):
    return (2 * I(v)**2 / (1-I(v)))**(1/3)

def t0(x, N):
    logN = np.log(N)
    return (x**2 + logN**2) / (2 * logN)

def function(x, N, chi1, chi2):
    t_vals = t0(x, N)
    exponent = t_vals**(1/3) * sigma(x / t_vals)
    return np.log(np.exp(exponent * chi1) + np.exp(exponent * chi2))

def var_power_long(x, N): 
    logN = np.log(N).astype(float)
    return 1/4 * np.sqrt(np.pi / 2) * x ** 3 / logN ** (5/2)

def var_power_short(x, N):
    logN = np.log(N).astype(float)
    return 0.8133 * x ** (8/3) / logN **(2) / 2 **(5/3)

def var_short(xvals, N, samples=10000):
    var = []
    tw = TracyWidom(beta=2)
    for x in xvals: 
        r1 = np.random.rand(samples)
        r2 = np.random.rand(samples)
        chi1 = tw.cdfinv(r1)
        chi2 = tw.cdfinv(r2)
        function_var = function(x, N, chi1, chi2)
        t_val = t0(x, N)
        prefactor = 1/(I(x / t_val) - x**2 / t_val ** 2 / np.sqrt(1 - (x/t_val)**2)) ** 2
        var.append(prefactor * np.var(function_var))

    return var 

def variance(x, N, samples=10000):
    crossover = (np.log(N).astype(float)) ** (3/2)
    width = (np.log(N).astype(float))**(4/3)
    theory_short = var_short(x, N, samples)
    theory_long =  var_power_long(x, N)
    error_func = (erf((x - crossover) / width) + 1) / 2
    theory = theory_short * (1 - error_func) + theory_long * (error_func)
    theory[x < np.log(N)] = 0
    return theory

def sam_variance_theory(x, N):
    t_vals = t0(x, N)
    beta = 1 / (I(x / t_vals) - x**2 / t_vals ** 2 / np.sqrt(1 - (x/t_vals)**2))
    return np.pi**2 * beta ** 2 / 6

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

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$x/ \log(N)$")
ax.set_xlim([0.5, 1000])
ax.set_ylabel(r"$\frac{\mathrm{Var}(\tau_{\mathrm{Env}})}{\sqrt{\log(N)}}$")
ax.plot(position7 / np.log(1e7), env_variance7 / np.sqrt(np.log(1e7)), c='m', label=r'$N=10^7$', alpha=0.5)

theory_pos = np.geomspace(np.log(1e7), 1000 * np.log(1e7), num=1000)
short = var_short(theory_pos, 1e7, samples=10000)
long= var_power_long(theory_pos, 1e7)
ax.plot(theory_pos / np.log(1e7), long / (np.sqrt(np.log(1e7))), label=r'KPZ Regime $N=10^{7}$', ls='--')
ax.plot(theory_pos / np.log(1e7), short / (np.sqrt(np.log(1e7))), label=r'TW Regime $N=10^{7}$', ls='--')
ax.vlines(np.log(1e7)**(3/2) / np.log(1e7), 0, 5*10**8, color='k', ls='--')
ax.set_ylim([10**-3, 5*10**8])
ax.annotate(r"$t=(\log(N))^{3/2}$", xy=(3, 10**5), rotation=90)

fig.savefig("EnvVarianceStitching.pdf", bbox_inches='tight')

fig, ax = plt.subplots()
ax.plot(position2 / np.log(1e2), env_mean2, c='b', label=r'$N=10^2$')
ax.plot(position7 / np.log(1e7), env_mean7, c='m', label=r'$N=10^7$')
ax.plot(position / np.log(1e24), env_mean24, c='r', label=r'$N=10^{24}$')

ax.set_xlim([10**-2, 10**3])
ax.set_ylim([1, 5*10**7])
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$x / log(N)$")
ax.set_ylabel(r"$Mean(\tau_{Env})$")
ax.legend()
fig.savefig("EnvMean.pdf", bbox_inches='tight')

fig, ax = plt.subplots()

alpha=0.5
colors = ['r', 'b', 'g']
fontsize=12

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$x/ \log(N)$")
ax.set_xlim([0.5, 1000])
ax.set_ylabel(r"$\mathrm{Var}(\tau_{\mathrm{Env}})$")
ax.plot(position2 / np.log(1e2), env_variance2, c=colors[0], label=r'$N=10^2$', alpha=alpha)
ax.plot(position7 / np.log(1e7), env_variance7, c=colors[1], label=r'$N=10^7$', alpha=alpha)
ax.plot(position / np.log(1e24), env_variance, c=colors[2], label=r'$N=10^{24}$', alpha=alpha)

for i, N in enumerate([1e2, 1e7, 1e24]):
    x = np.geomspace(1, 1000*np.log(N), num=2500)
    var = variance(x, N)
    ax.plot(x / np.log(N), var, c=colors[i], ls='--')

leg = ax.legend(
    fontsize=fontsize,
    loc="upper left",
    framealpha=0,
    labelcolor=colors,
    handlelength=0,
    handletextpad=0,
)

for item in leg.legendHandles:
    item.set_visible(False)

fig.savefig("EnvVariance.pdf", bbox_inches='tight')

N_exp = 24
N = float(f"1e{N_exp}")
logN = np.log(N).astype(float)
dir = "/home/jacob/Desktop/talapasMount/JacobData/FirstPassDiscreteAbs"
mean_file = os.path.join(dir, "Mean.txt")
var_file = os.path.join(dir, "Var.txt")
mean = np.loadtxt(mean_file).T
var = np.loadtxt(var_file).T

fig, ax = plt.subplots()
colors = ['b', 'g', 'r']
alpha = 0.75
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$x / \log(N)$")
ax.set_ylabel(r"$\mathrm{Variance}$")

ax.plot(position / logN, env_variance, c=colors[0], label=r'$\mathrm{Var}(\tau_{\mathrm{Env}})$', alpha=alpha)
ax.plot(position / logN, sam_variance, c=colors[1], label=r'$\mathrm{Var}(\tau_{\mathrm{Sam}})$', alpha=alpha)
ax.plot(var[:, 0] / logN, var[:, 1], c=colors[2], label=r'$\mathrm{Var}(\tau)$', alpha=alpha)

xvals = np.array([200, 750]) * logN
yvals3 = xvals ** 3 / 10**4 / 3
yvals4 = xvals ** 4 / 10**7

ax.plot(xvals / logN, yvals3, c='k', ls='--', label=r'$x^{3}$')
ax.plot(xvals / logN, yvals4, c='k', ls='--', label=r'$x^4$')

ax.set_ylim([10**-2, 10**12])
ax.set_xlim([0.5, 10**3])
leg = ax.legend(
    fontsize=fontsize,
    loc="upper left",
    framealpha=0,
    labelcolor=colors + ['k', 'k'],
    handlelength=0,
    handletextpad=0,
)

for item in leg.legendHandles:
    item.set_visible(False)
fig.savefig("TotalVariance.pdf", bbox_inches='tight')

sam_theoretical = sam_variance_theory(position[position > np.log(1e24)], 1e24)
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r'$x / \log(N)$')
ax.set_ylabel(r'$\mathrm{Var}(\tau_{\mathrm{Sam}})$')
ax.plot(position / np.log(1e24), sam_variance)
ax.plot(position[position > np.log(1e24)] / np.log(1e24), sam_theoretical, '--')
ax.set_ylim([10**-3, 10**12])
ax.set_xlim([0.5, 10**3])
fig.savefig("SamplingVariance.pdf", bbox_inches='tight')