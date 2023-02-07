import numpy as np
from matplotlib import pyplot as plt
from pyDiffusion.pydiffusion2D import getGCF1D
import time

dx = 0.5
D = 1
dt = dx**2 / 2 / D / 2
rc = 1
sigma = 1
L = 500

grid = np.arange(-L, L+dx, step=dx)
p0 = np.zeros(grid.size)
p0[p0.size // 2] = 1

fig, ax = plt.subplots()
ax.plot(grid, p0)
fig.savefig("InitialProb.png")

def forwardEuler(f, grid, dx, dt, D, rc, sigma):
    f_new = np.zeros(f.shape)
    field = getGCF1D(grid, rc, sigma)
    s = D * dt / dx**2
    v = dx / dt
    for i in range(1, len(f)-1):
        field_contributions = 1/v*(f[i] * field[i+1] + field[i] * f[i+1] - 2 * field[i] * f[i])
        f_new[i] = s * (f[i+1] + f[i-1]) + (1-2*s) * f[i] - field_contributions
    #print(sum(f_new))
    return f_new

start = time.time()
t = 0 
for i in range(50000):
    p0 = forwardEuler(p0, grid, dx, dt, D, rc, sigma)
    t += dt
    if i % 100 == 0:
        print(i, t, sum(p0))
print("Total Time:", time.time() - start)

def gaussian(x, mean, var):
    return 1 / np.sqrt(2 * np.pi * var) * np.exp(- (x-mean)**2 / 2 / var)

fig, ax = plt.subplots()
ax.set_yscale("log")
ax.plot(grid, p0)
ax.plot(grid, gaussian(grid, 0, 2 * D * t), c='k', ls='--', zorder=0, label='Gaussian')
ax.set_ylim([10**(-20), 1])
ax.set_ylabel(r"$\mathrm{Probability Density}$")
ax.set_xlabel(r"$x$")
ax.set_title(f"t={t}")
fig.savefig("ProbabilityDensity.pdf", bbox_inches='tight')