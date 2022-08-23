import numpy as np
import npquad
from matplotlib import pyplot as plt
import glob 
import pandas as pd

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
    mean = average_data / number_of_files
    variance = average_data_squared / number_of_files - mean ** 2
    return data[:, 0], mean, variance 
    

dir = "/home/jacob/Desktop/talapasMount/JacobData/FixedFirstPassCDF/24/F*.txt"
files = glob.glob(dir)
run_again = False

if run_again:
    position, mean, variance = calculateMeanVar(files, 27631)
    env_variance = variance[:, 1]
    sam_variance = mean[:, 2]
    np.savetxt("Position.txt", position)
    np.savetxt("Environmental.txt", env_variance)
    np.savetxt("SamplingVariance.txt", sam_variance)
else: 
    position = np.loadtxt("Position.txt")
    env_variance = np.loadtxt("Environmental.txt")
    sam_variance = np.loadtxt("SamplingVariance.txt")

theoretical_position = np.loadtxt("distances.txt")
theoretical_variance = np.loadtxt("varianceShortTime.txt")
theoretical_variance_long = np.loadtxt("varianceLongTime.txt")
theoretical_sampling = np.loadtxt("varianceSam.txt")

good_idx_short = (theoretical_variance <= theoretical_variance[-1])
good_idx = (theoretical_sampling > 10**-5) & (theoretical_sampling <= theoretical_sampling[-1])

N = np.quad("1e24")
logN = np.log(N).astype(float)

xvals = np.geomspace(5000, 27631)
yvals = 1/2 * np.sqrt(np.pi / 2) * xvals**3 / (logN)**(5/2)

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim([0.5, 500])
ax.set_ylim([10**-3, 10**11])
ax.set_xlabel(r"$x / \log(N)$")
ax.set_ylabel(r"$\mathrm{Var}(\tau)$")
ax.plot(position / logN, env_variance, c='r')
ax.plot(position / logN, sam_variance, c='b')
ax.plot(theoretical_position[good_idx_short] / logN, theoretical_variance[good_idx_short], c='k', ls='--')
ax.plot(theoretical_position / logN, theoretical_variance_long, c='m', ls='--')
ax.plot(theoretical_position[good_idx] / logN, theoretical_sampling[good_idx], c='k', ls='--')
# this is the long time asymptotics power law
# ax.plot(xvals / logN, yvals, c='orange')
fig.savefig("Variance.pdf", bbox_inches='tight')

dir = "/home/jacob/Desktop/talapasMount/JacobData/FixedFirstPassCDF/2/F*.txt"
files = glob.glob(dir)
run_again = False

if run_again:
    position2, mean, variance = calculateMeanVar(files, 2301)
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

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t/ \log(N)$")
ax.set_xlim([0.5, 500])
ax.set_ylabel(r"$\mathrm{Var}(\tau_{\mathrm{Env}})$")
ax.plot(position / np.log(1e24), env_variance, c='r')
ax.plot(position2 / np.log(float("1e2")), env_variance2, c='b')
ax.plot(theoretical_distance2 / np.log(100), theoretical_variance2, ls='--')
ax.plot(theoretical_distance2[2:] / np.log(100), theoretical_variance2_long, ls='--')
fig.savefig("EnvVariance.pdf", bbox_inches='tight')