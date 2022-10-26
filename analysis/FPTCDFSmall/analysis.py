import numpy as np
import pandas as pd
import glob
from matplotlib import pyplot as plt
import sys 
sys.path.append("../../dataAnalysis")
from fptTheory import variance, sam_variance_theory
plt.rcParams.update({"font.size": 15})

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

dir = '/home/jacob/Desktop/talapasMount/JacobData/FPTCDFSmall/F*.txt'
files = glob.glob(dir)
max_dist = 823
df, num_files = calculateMeanVarCDF(files, max_dist)

theory = variance(df['Distance'], 3+1)
xvals = np.array([100, 600])
yvals = xvals ** 3 / 10

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$L / \log(N)$")
ax.set_xlim([1, 750])
ax.set_ylabel(r"$\mathrm{Var}(\tau_{\mathrm{Env}})$")
ax.plot(df['Distance'] / np.log(3), df['Env Variance'], label='Data')
ax.plot(df['Distance'] / np.log(3), theory, ls='--', label='Theory(N=4)')
ax.plot(xvals, yvals, ls='--', c='k', label=r'$L^3$')
ax.legend()
fig.savefig("EnvVariance.pdf", bbox_inches='tight')

sam_theory = sam_variance_theory(df['Distance'], 3)

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$L / \log(N)$")
ax.set_xlim([1, 750])
ax.set_ylabel(r"$\mathrm{Var}(\tau_{\mathrm{Sam}})$")
ax.plot(df['Distance'] / np.log(3), df['Sampling Variance'], label='Data')
ax.plot(df['Distance'] / np.log(3), sam_theory, ls='--', label='Theory')
ax.plot(xvals, xvals ** 4 / 30, ls='--', c='k', label=r'$L^4$')
ax.legend()
fig.savefig("SamVariance.pdf", bbox_inches='tight')

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_xlabel(r"$L / \log(N)$")
ax.set_xlim([1, 750])
ax.set_ylabel(r"$\frac{\mathrm{Var}(\tau_{\mathrm{Env}})}{\mathrm{Var}(\tau_{\mathrm{Env}}^{\mathrm{Asym}})}}$")
ax.plot(df['Distance'] / np.log(3), df['Env Variance'] / theory)
fig.savefig("Ratio.pdf", bbox_inches='tight')
