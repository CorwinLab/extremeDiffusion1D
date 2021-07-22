import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 16})
import numpy as np
import glob
import os

files = glob.glob('/home/jhass2/Data/1.0/TracyWidom/T*.txt')
with open(files[0]) as g: 
    vs = g.readline().split(',')[1:]
    vs = [float(i) for i in vs]

if not os.path.exists('/home/jhass2/Data/1.0/TracyWidom/mean.txt'):
    squared_sum = None
    reg_sum = None

    for f in files:
        data = np.loadtxt(f, delimiter=',', skiprows=1)
        time = data[:, 0]
        data = data[:, 1:]
        data = np.log(data/1e300) # subtract of N=1e300

        if squared_sum is None:
            squared_sum = data ** 2
        else:
            squared_sum += data ** 2

        if reg_sum is None:
            reg_sum = data
        else:
            reg_sum += data

    mean = reg_sum / len(files)
    var = squared_sum / len(files) - mean ** 2
    np.savetxt('/home/jhass2/Data/1.0/TracyWidom/mean.txt', mean)
    np.savetxt('/home/jhass2/Data/1.0/TracyWidom/var.txt', var)
else:
    mean = np.loadtxt('/home/jhass2/Data/1.0/TracyWidom/mean.txt')
    var = np.loadtxt('/home/jhass2/Data/1.0/TracyWidom/var.txt')
    data = np.loadtxt(files[0], delimiter=',', skiprows=1)
    time = data[:, 0]

for i in range(len(vs)):
    v = vs[i]
    v_var = var[:, i]
    I = 1 - np.sqrt(1-v**2)
    sigma = (2 * I**2 / (1-I)) ** (1/3)
    theory = time**(2/3) * sigma**2
    fig, ax = plt.subplots()
    ax.set_xlabel('Time')
    ax.set_ylabel(r'$Var(Ln(P_{b}(vt, t)))$')
    ax.plot(time, v_var, c='b', label='Data')
    ax.plot(time, theory, c='k', label=r'$t^{2/3}\sigma^{2}$')
    ax.set_title(f'v={v} & {len(files)} Systems')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    fig.savefig(f'./figures/Variance{v}.png', bbox_inches='tight')
    plt.close(fig)

for i in range(len(vs)):
    v = vs[i]
    I = 1 - np.sqrt(1-v**2)
    sigma = (2 * I**2 / (1-I)) ** (1/3)
    theory = -I * time + time**(1/3) * sigma * -1.77
    fig, ax = plt.subplots()
    ax.set_xlabel('Time')
    ax.set_ylabel(r'$|Ln(P_{b}(vt, t))|$')
    ax.plot(time, abs(theory), c='k', label=r'$-I*t - 1.77*t^{1/3}*\sigma$')
    ax.plot(time, abs(mean[:, i]), c='b', label='Data')
    ax.set_title(f'v={v} & {len(files)} Systems')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    fig.savefig(f'./figures/Mean{v}.png', bbox_inches='tight')
    plt.close(fig)
