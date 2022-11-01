import numpy as np
from matplotlib import pyplot as plt
import glob 
import pandas as pd

dir = '/home/jacob/Desktop/corwinLabMount/CleanData/ContinuousLargeField/M*.txt'
files = glob.glob(dir)
data_sum = None
data_sum_squared = None
maxTime = 1500
num_files = 0
for f in files: 
    try:
        data = pd.read_csv(f)
    except:
        continue
    if max(data['Time']) < maxTime:
        continue 
    data = data[data['Time'] <= maxTime]
    time = data['Time'].values
    if data_sum is None: 
        data_sum = data['Position'].values 
    else: 
        data_sum += data['Position'].values
    if data_sum_squared is None:
        data_sum_squared = (data['Position'].values) ** 2
    else:
        data_sum_squared += (data['Position'].values)**2
    num_files += 1

mean = data_sum / num_files
var = data_sum_squared / num_files - mean**2 
np.savetxt('Time.txt', time)
np.savetxt("Mean.txt", mean)
np.savetxt("Var.txt", var)
print(f"Number of files: {num_files}")

fig, ax = plt.subplots()
xvals = np.linspace(100, 1000)
#ax.plot(xvals, 4*xvals ** (1/2), ls='--', label=r'$\sqrt{t}$', c='k')
ax.plot(time, mean, c='b', label=r'$\xi_1$')
ax.set_xlim([1, 1500])
ax.set_xlabel("Time")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylabel("Mean")
ax.legend()
fig.savefig("Mean.png", bbox_inches='tight')

fig, ax = plt.subplots()
#ax.plot(xvals, xvals/10, ls='--', c='k', label=r'$t$')
ax.plot(time, var, c='b',  label=r'$\xi_1$')
ax.set_xlim([1, 1500])
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Time")
ax.set_ylabel("Variance")
ax.legend()
fig.savefig("Var.png", bbox_inches='tight')


    