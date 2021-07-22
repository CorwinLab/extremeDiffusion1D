import numpy as np
from matplotlib import pyplot as plt
import glob
import os

files = glob.glob('./Data/D*.txt')
data = np.loadtxt('./Data/Data0.txt', max_rows=10_000)
time = data[:,0]
sum = None
squared_sum = None

if not os.path.isfile('./Data/mean.txt'):
    for f in files:
        data = np.loadtxt(f, max_rows=10_000)
        time = data[:, 0]
        data = 2 * data[:, 1]
        if sum is None:
            sum = data
        else:
            sum += data

        if squared_sum is None:
            squared_sum = data ** 2
        else:
            squared_sum += data ** 2
        print(f)

    mean = sum / len(files)
    var = squared_sum / len(files) - mean ** 2
    np.savetxt('./Data/mean.txt', mean)
    np.savetxt('./Data/var.txt', var)

else:
    mean = np.loadtxt('./Data/mean.txt')
    var = np.loadtxt('./Data/var.txt')

fig, ax = plt.subplots()
ax.set_xlabel('Time')
ax.set_ylabel('Mean Displacement')
ax.plot(time, mean)
ax.set_xlim([0, 10_000])
fig.savefig('Mean.png')

fig, ax = plt.subplots()
ax.set_xlabel('Time')
ax.set_ylabel('Variance')
ax.plot(time, var)
ax.grid(True)
ax.set_xlim([0, 10_000])
ax.set_ylim([0, 10_000])
fig.savefig('Variance.png')
