import numpy as np
import npquad
from matplotlib import pyplot as plt
import glob 
import pandas as pd
import os

def calculateMeanVar(files, max_dist, verbose=True):
    average_data = None
    average_data_squared = None
    number_of_files = 0
    for f in files:
        data = pd.read_csv(f, delimiter=',') # columns are position, quantile, variance
        
        if max(data['Position']) < max_dist: 
            continue
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
    np.savetxt("Position.txt", position)
    np.savetxt("Environmental.txt", env_variance)
    np.savetxt("SamplingVariance.txt", sam_variance)
else: 
    position = np.loadtxt("Position.txt")
    env_variance = np.loadtxt("Environmental.txt")
    sam_variance = np.loadtxt("SamplingVariance.txt")


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

theoretical_position = np.loadtxt("distances.txt")
theoretical_variance = np.loadtxt("varianceShortTime.txt")
theoretical_variance_long = np.loadtxt("varianceLongTime.txt")
theoretical_sampling = np.loadtxt("varianceSam.txt")

distanceMax = np.loadtxt("../FirstPassDiscreteAbs/distances.txt")
varianceMax = np.loadtxt("../FirstPassDiscreteAbs/varianceMax.txt")
good_idx = (varianceMax > 10**-3)  * (varianceMax < varianceMax[-1])
varianceMax = varianceMax[good_idx]
distanceMax = distanceMax[good_idx]

good_idx_short = (theoretical_variance <= theoretical_variance[-1])
good_idx = (theoretical_sampling > 10**-5) & (theoretical_sampling <= theoretical_sampling[-1])

N = np.quad("1e24")
logN = np.log(N).astype(float)

# check asymptotic power laws
xvals = np.geomspace(5000, 27631)
yvals = 1/2 * np.sqrt(np.pi / 2) * xvals**3 / (logN)**(5/2)
ysam = np.pi**2 / 24 * xvals**4 / (logN)**4

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
'''
ax.plot(theoretical_position[good_idx_short] / logN, theoretical_variance[good_idx_short], c='k', ls='--')
ax.plot(theoretical_position / logN, theoretical_variance_long, c='m', ls='--')
ax.plot(theoretical_position[good_idx] / logN, theoretical_sampling[good_idx], c='k', ls='--')
'''
# this is the long time asymptotics power law
# ax.plot(xvals / logN, yvals, c='orange')
#ax.plot(xvals / logN, ysam, c='orange')
fig.savefig("Variance.pdf", bbox_inches='tight')

dir = "/home/jacob/Desktop/talapasMount/JacobData/FixedFirstPassCDF/2/F*.txt"
files = glob.glob(dir)
run_again = False

if run_again:
    position2, mean, variance = calculateMeanVar(files, 4605)
    env_variance2 = variance[:, 1]
    sam_variance2 = mean[:, 2]
    np.savetxt("Position2.txt", position2)
    np.savetxt("Environmental2.txt", env_variance2)
    np.savetxt("SamplingVariance2.txt", sam_variance2)
else: 
    position2 = np.loadtxt("Position2.txt")
    env_variance2 = np.loadtxt("Environmental2.txt")
    sam_variance2 = np.loadtxt("SamplingVariance2.txt")

theoretical_distance2 = np.loadtxt("distances2.txt")
theoretical_variance2 = np.loadtxt("varianceShortTime2.txt")
theoretical_variance2_long = np.loadtxt("varianceLongTime2.txt")

theoretical_distance7 = np.loadtxt("distances7.txt")
theoretical_variance7 = np.loadtxt("varianceShortTime7.txt")
theoretical_variance7_long = np.loadtxt("varianceLongTime7.txt")

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$x/ \log(N)$")
ax.set_xlim([0.5, 1000])
ax.set_ylabel(r"$\frac{\mathrm{Var}(\tau_{\mathrm{Env}})}{\sqrt{\log(N)}}$")
ax.plot(position / np.log(1e24), env_variance/(np.sqrt(np.log(1e24))), c='r', label=r'$N=10^{24}$', alpha=0.5)
ax.plot(position2 / np.log(1e2), env_variance2/(np.sqrt(np.log(1e2))), c='b', label=r'$N=10^2$', alpha=0.5)
ax.plot(position7 / np.log(1e7), env_variance7 / np.sqrt(np.log(1e7)), c='m', label=r'$N=10^7$', alpha=0.5)
ax.plot(theoretical_distance2 / np.log(100), theoretical_variance2_long / np.sqrt(np.log(1e2)), ls='-.', c='b')
ax.plot(theoretical_distance7 / np.log(1e7), theoretical_variance7_long / np.sqrt(np.log(1e7)), ls='-.', c='m')
ax.plot(theoretical_distance7 / np.log(1e7), theoretical_variance7 / np.sqrt(np.log(1e7)), ls='--', c='m')
ax.plot(theoretical_distance2 / np.log(1e2), theoretical_variance2 / np.sqrt(np.log(1e2)), ls='--', c='b')

# Plotting the 1/N quantile stuff
distances = [20, 100, 1611]
for d in distances:
    data = pd.read_csv(f"../FirstPassTest/TotalDF{d}.csv")
    data_first_pass = data[data['Number Crossed'] == 0]

    #ax.scatter(d / np.log(1e7), np.var(data['Time'].values), c='g')
    #ax.scatter(distance / np.log(1e7), np.var(data_first_pass['Time'].values), c='r')
minTimes = np.loadtxt("../DoubleSidedFPT/MinimumTimes.txt")
#ax.scatter(100 / np.log(1e7), np.var(minTimes), c='k')
ax.legend()
fig.savefig("EnvVariance.pdf", bbox_inches='tight')

logN = np.log(1e7)
fig, ax = plt.subplots()
ax.plot(position7 / logN, env_mean7, c='m', label='Data')
ax.plot(position7[position7 > logN] / logN, position7[position7 >logN]**2 / 2 /logN, '--', c='orange', label='Theory by Inversion')

distances = [20, 100, 1611]
for d in distances:
    data = pd.read_csv(f"../FirstPassTest/TotalDF{d}.csv")
    data_first_pass = data[data['Number Crossed'] == 0]

    #ax.scatter(d / np.log(1e7), np.mean(data['Time'].values), c='g')

ax.scatter(100 / np.log(1e7), np.mean(minTimes), c='k')

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$x / log(N)$")
ax.set_ylabel(r"$Mean(\tau_{Env})$")
ax.legend()
fig.savefig("EnvMean.pdf", bbox_inches='tight')