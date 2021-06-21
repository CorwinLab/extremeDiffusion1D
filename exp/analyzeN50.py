import numpy as np
from matplotlib import pyplot as plt
import glob

files = glob.glob('./N50/*')
fig, ax = plt.subplots()
ax.set_xlabel("Time")
ax.set_ylabel("Distance from Origin")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_title("N=1e50, alpha=beta=1")
ax.grid(True)

'''
all_data = None

for file in files:
    data = np.loadtxt(file)
    steps = np.arange(1, len(data)+1) * 0.5
    if all_data is None:
        all_data = data
    else:
        all_data = np.vstack((all_data, data))
    print(file)

mean = np.mean(all_data, 0)
var = np.var(all_data, 0)
np.savetxt('mean.txt', mean)
np.savetxt('variance.txt', var)
'''
mean = np.loadtxt('mean.txt')
var = np.loadtxt('variance.txt')
steps = np.arange(1, len(mean)+1) * 0.5

ax.plot(steps, mean, c='k', label='mean')
ax.fill_between(steps, mean-3*var, mean+3*var, color='b', alpha=0.5, label='three std')
ax.legend()
fig.savefig("distanceN50.png")

fig, ax = plt.subplots()
ax.plot(steps, var, c='k')
ax.set_xlabel('Time')
ax.set_ylabel('Variance')
ax.set_title('N=1e50, alpha=beta=1')
ax.set_xscale('log')
ax.set_yscale('log')
fig.savefig('var.png')
