import matplotlib 
matplotlib.use('Agg')
import numpy as np
from matplotlib import pyplot as plt
import glob

files = glob.glob('/home/jhass2/Data/1.0/1/Q*.txt')
mean = np.loadtxt('/home/jhass2/Data/1.0/1/mean.txt')
var = np.loadtxt('/home/jhass2/Data/1.0/1/var.txt')
data = np.loadtxt('/home/jhass2/Data/1.0/1/Quartiles1.txt')
time = data[:, 0]

Ns = np.geomspace(1e10, 1e50, 9)
N = Ns[0]

# Just plot the mean versus the theoretically predicted curve
fig, ax = plt.subplots()
ax.set_xlabel('Time/Log(N)')
ax.set_ylabel('Mean Quantile')
ax.set_title(f'Mean 1/Nth Quantile for N=1e10 & {len(files)} Systems')
ax.set_xscale('log')
ax.set_yscale('log')
theory = np.piecewise(time, [time < np.log(N), time >= np.log(N)], [lambda x: x, lambda x: x * np.sqrt(1 - (1-np.log(N)/x)**2)])
ax.plot(time/np.log(N), theory)
ax.plot(time/np.log(N), mean[:,0])
fig.savefig('./figuresBump/MeanQuartile10.png', bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots()
cm = plt.get_cmap('gist_heat')
ax.set_color_cycle([cm(1. * i /len(Ns)/1.5) for i in range(len(Ns))])
for i, N in enumerate(Ns): 
    theory = np.piecewise(time, [time < np.log(N), time >= np.log(N)], [lambda x: x, lambda x: x * np.sqrt(1-(1-np.log(N)/x)**2)])
    ax.set_xlabel('Time/Log(N)')
    ax.set_ylabel('1 - Mean Quantile / Time')
    ax.set_title(f'Mean 1/Nth Quantile for N=1e10-50 & {len(files)} Systems')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot(time / np.log(N), 1 - mean[:,i] / time, label=None)
ax.plot(time / np.log(N), 1 - theory / time, label='Theory', c='b')
ax.legend()
fig.savefig('./figuresBump/MeanDivTime10.png', bbox_inches='tight')
plt.close(fig)
