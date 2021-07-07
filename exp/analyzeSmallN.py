import matplotlib
matplotlib.use('Agg')
import numpy as np
from matplotlib import pyplot as plt

data = np.loadtxt('Data.txt')
times = data[:,0]
maxEdge = data[:,1]
Ns = np.geomspace(1e20, 1e280, 14)

fig, ax = plt.subplots()
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Time')
ax.set_ylabel('Nth Quartile')

for i, N in enumerate(Ns[:2]):
    Nstr = "{:.0e}".format(N)
    theory = np.piecewise(times, [times < np.log(N), times >= np.log(N)], [lambda x: x, lambda x: x*np.sqrt(1 - (1-np.log(N)/x)**2)])
    ax.plot(times+1, 2 * data[:, i+2], label='N='+Nstr + ' Data')
    ax.plot(times, theory, label='Theoretical Curve')

ax.legend()
fig.savefig('thoery.png')
