import glob
import sys
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def calculateMeanVarCDF(files, max_dist, verbose=True, nFile=np.inf):
    # Calculate mean and variance
    average_data = None
    average_data_squared = None
    number_of_files = 0
    pos = None
    for f in files: 
        try: 
            data = pd.read_csv(f) # columns are Position, Quantile, Variance
        except pd.io.common.EmptyDataError:
            print(f"Empty file: {f}")
            continue
        
        if max(data['Radius']) < max_dist:
            print("Not Enough Data: ", f, max(data['Radius']))
            continue
        data = data[data['Radius'] <= max_dist]
        data = data.values
        pos = data[:, 0]
        number_of_files += 1
        if average_data is None:
            average_data = data
            average_data_squared = data ** 2
        else:
            average_data += data
            average_data_squared += data ** 2
        if number_of_files >= nFile:
            break

        if verbose:
            print(f, max(data[:, 0]))
            
    if number_of_files == 0:
        return None
    
    mean = average_data / number_of_files
    variance = average_data_squared / number_of_files - mean ** 2

    # Calculate variance of env variance
    forth_moment = None
    forth_moment_files = 0
    for f in files:
        data = pd.read_csv(f) # columns are Position, Quantile, Variance
        if max(data['Radius']) < max_dist:
            continue

        data = data[data['Radius'] <= max_dist]
        data = data.values
        pos = data[:, 0]
        forth_moment_files += 1

        if forth_moment is None:
            forth_moment = (data- mean) ** 4
        else: 
            forth_moment += (data - mean) ** 4
        
        if forth_moment_files >= nFile:
            break


    forth_moment = (forth_moment / forth_moment_files - variance ** 2) / forth_moment_files
    new_df = pd.DataFrame(np.array([pos, mean[:, 1], mean[:, 2], variance[:, 1], variance[:, 2], forth_moment[:, 1]]).T, columns=['Radius', 'Mean Quantile', 'Sampling Variance', 'Env Variance', 'Var Sampling Variance', 'Var Env Variance'])
    return new_df, number_of_files

files = glob.glob("/home/jacob/Desktop/talapasMount/JacobData/2DLatticeRWRESpherical/F*.txt")
max_dist = 210

df, num = calculateMeanVarCDF(files, max_dist, verbose=False)
logN = np.log(1e24)

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$L / \log(N)$")
ax.set_ylabel(r"$\mathrm{Mean}(\mathrm{Env}_t^N)$")
ax.plot(df['Radius'] / logN, df['Mean Quantile'])

xvals = np.array([3, 7])
ax.plot(xvals, xvals**2*50, c='k', ls='--', label=r'$L^2$')

xvals = np.array([0.1, 0.7])
ax.plot(xvals, xvals*50, c='grey', ls='--', label=r'$L$')

ax.set_xlim([min(df['Radius']) / logN, max(df['Radius'])/logN])
ax.legend()
fig.savefig("Mean.pdf", bbox_inches='tight')

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$L / \log(N)$")
ax.set_ylabel("Variance")
ax.plot(df['Radius'] / logN, df['Env Variance'], label=r"$\mathrm{Var}(\mathrm{Env}_t^N)$")

ax.plot(df['Radius'] / logN, df['Sampling Variance'], label=r"$\mathrm{Var}(\mathrm{Sam}_t^N)$")

xvals = np.array([2, 3])
ax.plot(xvals, xvals**4 * 2, c='k', ls='--', label=r'$L^4$')

#ax.plot(xvals, xvals**(5/3) * 3, c='r', ls='--', label=r'$L^{5/3}$')

ax.set_xlim([min(df['Radius']) / logN, max(df['Radius'])/logN])
ax.set_ylim([10**-2, 10**3])
ax.legend()
fig.savefig("Variance.pdf", bbox_inches='tight')