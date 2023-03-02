import numpy as np
from matplotlib import pyplot as plt
import time
from numba import njit

@njit
def generateGCF1D(pos, xi, tol=0.01):
	"""
	Parameters
	----------
	pos : numpy array
		Position of particles. Array should have: rows = particleID and
		cols = components

	xi : float
		Correlation length

	fourierCutoff : int (20)
		Number of terms to include in fourier transform.

	Returns
	-------
	c : numpy array 
		Gaussian correlated field with shape rows = particleID and 
		cols = components

	Examples
	--------
	"""
	
	_pos = pos.copy()
	_pos = (_pos - np.min(_pos)) # need to account for negative minimum value
	
	L = (np.max(_pos) - np.min(_pos)) + 3 * xi
	fourierCutoff = int(np.sqrt(-L**2 * 4 * np.log(tol)/(2 * np.pi)**2))

	# Coerce all particles to be in the box from [0, 1]
	_pos = _pos / L
	xi = xi / L
	#theta = np.random.uniform(0, 2*np.pi)

	field = np.zeros(_pos.shape)
	A = np.random.normal(0, 1/np.sqrt(2*np.pi), size=(2*fourierCutoff))
	B = np.random.uniform(0, 2 * np.pi, size=(2*fourierCutoff))
	
	for pID in range(len(_pos)):
		for n in np.arange(-fourierCutoff, fourierCutoff):
			kn = 2 * np.pi * n
			# This could shift the two point correlator to something we don't want 
			#xrot = np.cos(theta) * pos[pID, 0] - np.sin(theta) * pos[pID, 1]
			#yrot = np.sin(theta) * pos[pID, 0] + np.cos(theta) * pos[pID, 1]
			field[pID] += np.sqrt(2) * np.pi * A[n+fourierCutoff] * np.exp(-kn**2 * xi**2 / 8) * np.cos(B[n+fourierCutoff] + kn * _pos[pID])

	return field / np.sqrt(L)

dx = 0.1
D = 1
dt = dx**2 / 2 / D / 4
rc = 1
sigma = 1
L = 5

grid = np.arange(-L, L+dx, step=dx)
p0 = np.zeros(grid.size)
p0[p0.size // 2] = 1

fig, ax = plt.subplots()
ax.plot(grid, p0)
fig.savefig("InitialProb.png")

def forwardEuler(f, grid, dx, dt, D, rc, tol):
    f_new = np.zeros(f.shape)
    field = generateGCF1D(grid, rc, tol)
    for i in range(1, len(f)-1):
        field_times_prob = f * field
        f_new[i] = (D / dx**2 * (f[i+1] - 2 * f[i] + f[i-1]) - 1/(2*dx) * (field_times_prob[i+1] - field_times_prob[i-1])) * dt + f[i]
    return f_new

start = time.time()
t = 0 
for i in range(1000):
    p0 = forwardEuler(p0, grid, dx, dt, D, rc, 0.001)
    if np.sum(p0 < 0) != 0:
        print(f"Less than zero = {np.sum(p0 <0)}")
    t += dt
    if i % 100 == 0:
        print(i, t, sum(p0))
        
print("Total Time:", time.time() - start)

def gaussian(x, mean, var):
    return 1 / np.sqrt(2 * np.pi * var) * np.exp(- (x-mean)**2 / 2 / var)

fig, ax = plt.subplots()
ax.set_yscale("log")
ax.plot(grid, p0 / dx)
ax.plot(grid, gaussian(grid, 0, 2 * D * t), c='k', ls='--', zorder=0, label='Gaussian')
ax.set_ylim([10**(-20), 1])
ax.set_ylabel(r"$\mathrm{Probability Density}$")
ax.set_xlabel(r"$x$")
ax.set_title(f"t={t}")
fig.savefig("ProbabilityDensity.pdf", bbox_inches='tight')