import matplotlib
matplotlib.use('Agg')
import numpy as np
from matplotlib import pyplot as plt
import glob
import os

files = glob.glob('/home/jhass2/Data/1.0/1Large/Q*.txt')
Ns = [float(10**i) for i in range(20, 300, 20)]
running_sum = None
running_sum_squared = None
count = 0
if (not os.path.isfile('/home/jhass2/Data/1.0/1Large/mean.txt')) or True:
    for f in files:
        basename = os.path.basename(f)
        name = basename.split('.')[0]
        num = name.replace('Quartiles', '')
        if int(num) > 89:
            continue
        data = np.loadtxt(f, delimiter=',', skiprows=1)
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
        count += 1

    mean = running_sum / count
    var = running_sum_squared / count - mean ** 2 

    np.savetxt('/home/jhass2/Data/1.0/1Large/mean.txt', mean)
    np.savetxt('/home/jhass2/Data/1.0/1Large/var.txt', var)

else: 
    mean = np.loadtxt('/home/jhass2/Data/1.0/1Large/mean.txt')
    var = np.loadtxt('/home/jhass2/Data/1.0/1Large/var.txt')

fig, ax = plt.subplots()
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Time/Log(N)')
ax.set_ylabel('Var of Nth Quartile/Log(N)^(2/3)')
ax.set_title('Variance of Nth Quartile')
cm = plt.get_cmap('gist_heat')
ax.set_color_cycle([cm(1. * i / len(Ns)/1.5) for i in range(len(Ns))])

for i in range(len(Ns)):
    N = Ns[i]
    Nstr = '{:.0e}'.format(N)
    ax.plot(times / np.log(N), var[:, i] / (np.log(N) ** (2/3)), label='N='+Nstr)
ax.legend()
fig.savefig('./figuresVar/Variance.png')

fig, ax = plt.subplots()
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Time / Log(N)')
ax.set_ylabel('Var of Nth Quartile * (Log(N)/Time)^(1/3)')
ax.set_title('Variance of Nth Quartile')
cm = plt.get_cmap('gist_heat')
ax.set_color_cycle([cm(1.0 * i / len(Ns) / 1.5) for i in range(len(Ns))])

scale = 1/3
scale = (times / np.log(N)) ** scale

for i in range(len(Ns)):
    N = Ns[i] 
    Nstr = '{:.0e}'.format(N)
    ax.plot(times / np.log(N), var[:, i] / (np.log(N) ** (2/3)) / scale, label='N='+Nstr)
ax.legend()
fig.savefig('./figuresVar/VarianceScaled.png')
