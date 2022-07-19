import os 
import glob 
import sys 
sys.path.append("../../src")
import numpy as np
import npquad
from matplotlib import pyplot as plt
from theory import quantileVar

dir = "/home/jacob/Desktop/talapasMount/JacobData/FirstPassCDF2/F*.txt"
files = glob.glob(dir)
average_data = None
average_data_squared = None
number_of_files = 0
for f in files:
    data = np.loadtxt(f, skiprows=1, delimiter=',') # columns are distance, mean, variance, quantile position
    print(f)
    if data[-1, 0] != 499.0: 
        continue
    number_of_files += 1
    if average_data is None: 
        average_data = data
        average_data_squared = data ** 2
    else:
        average_data += data
        average_data_squared += data ** 2

average_data = average_data / number_of_files
quantile_position_var = average_data_squared[:, 3] / number_of_files - average_data[:, 3]**2
average_data = np.hstack((average_data, quantile_position_var.reshape(average_data.shape[0], 1)))
# columns are distance, N particle mean, N particle variance, mean quantile, variance of quantile
np.savetxt("AveragedData.txt", average_data)

N = np.quad("1e24")
logN = np.log(N).astype(float)
distance = average_data[:, 0]

'''Plot the mean first passage time'''
fig, ax = plt.subplots()
ax.set_xlabel("Distance / log(N)")
ax.set_ylabel("Mean(First Passage Time)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.plot(distance / logN, average_data[:,1], label='Data')
xvals = np.array([3*10**2, 4*10**2]) / logN
yvals = np.array([3*10**2, 4*10**2])**2/200
ax.plot(xvals, yvals, c='k', label=r'$t^{2}$')
ax.set_xlim([min(distance) / logN, max(distance)/logN])
ax.legend()
fig.savefig("Mean.png")

'''Plot the variance of the first passage time'''
fig, ax = plt.subplots()
ax.set_xlabel("Distance / log(N)")
ax.set_ylabel("Var(First Passage Time)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.plot(average_data[:, 0] / logN, average_data[:, 2], label='Data')
xvals = np.array([3*10**2, 4*10**2]) / logN
yvals = np.array([3*10**2, 4*10**2])**4/10**8
ax.plot(xvals, yvals, c='k', label=r'$t^{4}$')
ax.set_xlim([min(distance) / logN, max(distance)/logN])
ax.legend()
fig.savefig("Variance.png")