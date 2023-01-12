import numpy as np
from matplotlib import pyplot as plt
from pyDiffusion.pydiffusion2D import getGCF1D

dx = 0.01
D = 1
dt = dx**2 / 2 / D / 2

grid = np.arange(-100, 100+dx, step=dx)
p0 = np.zeros(grid.size)
p0[p0.size // 2] = 1

fig, ax = plt.subplots()
ax.plot(grid, p0)
fig.savefig("InitialProb.png")

def forwardEuler(f, grid, dx, dt, D):
    f_new = np.zeros(f.shape)
    field = getGCF1D(grid, 0.25, D)
    s = D * dt / dx**2
    v = dx / dt
    for i in range(1, len(f)-1):
        field_contributions = 1/v*(f[i] * field[i+1] + field[i] * f[i+1] - 2 * field[i] * f[i])
        f_new[i] = s * (f[i+1] + f[i-1]) + (1-2*s) * f[i] - field_contributions
    print(sum(f_new))
    return f_new

for i in range(10000):
    p0 = forwardEuler(p0, grid, dx, dt, D)
    print(i)

fig, ax = plt.subplots()
ax.set_yscale("log")
ax.plot(grid, p0)
fig.savefig("NextTimeStep.png")