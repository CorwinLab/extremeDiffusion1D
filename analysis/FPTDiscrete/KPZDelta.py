import numpy as np
from matplotlib import pyplot as plt

N = 1e12
x = np.geomspace(np.log(N), 1000*np.log(N))

def delta(L, N):
    logN = np.log(N)
    prefactor = 1 / (2 * logN / L**2 - 2 *logN**2 / L**2 - 4 * logN**4/L**4)
    return prefactor * (-logN**3/3/L**2 - np.log(np.sqrt(16*np.pi*logN**3/L**2)) +np.log(2) + np.log(2*logN/L)-2*logN**3/3/L**2)

def t0(L, N):
    return L**2/2/np.log(N)

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylabel("Mean")
ax.set_xlabel(r"$L/\log(N)$")
ax.plot(x/np.log(N), delta(x, N),label=r'$\delta$')
ax.plot(x/np.log(N), t0(x, N), label=r'$t_0$')
ax.set_xlim(1, 1000)
ax.legend()
ax.grid(True)
fig.savefig("KPZMean.pdf", bbox_inches='tight')