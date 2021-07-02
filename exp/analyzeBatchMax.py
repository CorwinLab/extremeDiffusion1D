import matplotlib
matplotlib.use('Agg')
import numpy as np
from matplotlib import pyplot as plt
import glob
import os

files = glob.glob('/home/jhass2/Data/1.0/1/Q*.txt')
data = np.loadtxt('/home/jhass2/Data/1.0/1/Quartiles1.txt')
times = data[:, 0]
Ns = np.geomspace(1e10, 1e50, 9)
running_sum = None
running_sum_squared = None

if not os.path.isfile('/home/jhass2/Data/1.0/1/mean.txt'):
    for f in files:
        data = np.loadtxt(f)
        times = data[:, 0]
        maxEdge = data[:, 1]
        data = data[:, 2:]
        if running_sum is None: 
            running_sum = 2 * data
        else:
            running_sum += 2 * data

        if running_sum_squared is None:
            running_sum_squared = (2*data) ** 2 
        else:
            running_sum_squared += (2*data) ** 2

    mean = running_sum / len(files)
    var = running_sum_squared / len(files) - mean ** 2 

    np.savetxt('/home/jhass2/Data/1.0/1/mean.txt', mean)
    np.savetxt('/home/jhass2/Data/1.0/1/var.txt', var)

else: 
    mean = np.loadtxt('/home/jhass2/Data/1.0/1/mean.txt')
    var = np.loadtxt('/home/jhass2/Data/1.0/1/var.txt')

fig, ax = plt.subplots()
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Time/Log(N)')
ax.set_ylabel('Nth Quartile')
ax.set_title(f'Number of Systems = {len(files)}')

for i in range(len(Ns)):
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Time/Log(N)')
    ax.set_ylabel('Mean Nth Quartile')
    ax.set_title('Mean Nth Quartile versus Theory')
    Nstr = '{:.0e}'.format(Ns[i])
    ax.plot(times/np.log(Ns[i]), mean[:, i], label='N='+Nstr + 'Data')
    theory = np.piecewise(times, [times < np.log(Ns[i]), times >= np.log(Ns[i])], [lambda x: x, lambda x: x*np.sqrt(1-(1-np.log(Ns[i])/x)**2)])
    ax.plot(times/np.log(Ns[i]), theory, c='k', label='Theoretical Curve')

    ax.legend()
    fig.savefig('./figures/meanQuartile'+Nstr+'.png')

fig, ax = plt.subplots()
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Time/Log(N)')
ax.set_ylabel('Variance of Nth Quartile/Log(N)^(2/3)')
ax.set_title(f'Number of Systems = {len(files)}')

for i in range(len(Ns)):
    Nstr = '{:.0e}'.format(Ns[i])
    ax.plot(times/np.log(Ns[i]), var[:, i]/(np.log(Ns[i])**(2/3)), label='N='+Nstr)
    N = Ns[i]
    theory = (2*np.log(N))**(2/3) * (times/np.log(N)-1)**(4/3) / (2*times/np.log(N)-1)
    ax.plot(times/np.log(N), theory/(np.log(N) **(2/3)))

ax.legend()
fig.savefig('./figures/varQuartile.png')

fig, ax = plt.subplots()
ax.set_xlabel('Time/Log(N)')
ax.set_ylabel('Residual')
ax.set_title(f'Number of Systems = {len(files)}')
ax.set_xscale('log')
ax.set_yscale('log')

for i in range(len(Ns)):
    N = Ns[i]
    Nstr = '{:.0e}'.format(N)
    theory = np.piecewise(times, [times < np.log(N), times >= np.log(N)], [lambda x: x, lambda x: x*np.sqrt(1-(1-np.log(N)/x)**2)])
    residual = theory - mean[:, i]
    ax.plot(times/np.log(N), residual, label='N='+Nstr)

ax.legend()
fig.savefig('./figures/residualQuartile.png')
