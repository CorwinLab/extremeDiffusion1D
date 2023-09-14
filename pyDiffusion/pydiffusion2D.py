import numpy as np
from numba import njit 
import csv

@njit
def meshgrid(x, y):
	xx = np.empty(shape=(x.size, y.size), dtype=x.dtype)
	yy = np.empty(shape=(x.size, y.size), dtype=y.dtype)
	for j in range(y.size):
		for k in range(x.size):
			xx[j,k] = x[j]  
			yy[j,k] = y[k]
	return yy, xx

@njit
def generateGCF2D(pos, xi, sigma, tol=0.01):
	_pos = pos.copy()
	num_particles, dims = _pos.shape

	Lx = (np.max(_pos[:,0]) - np.min(_pos[:,0])) + 3 * xi
	Ly = (np.max(_pos[:,1]) - np.min(_pos[:,1])) + 3 * xi
	L = np.max(np.array([Lx, Ly]))

	# Coerce all particles to be in the box from [0, 1]
	_pos[:, 0] /= L # need to account for minimum values! (i.e. if the minimum is not 0)
	_pos[:, 1] /= L
	xi /= L
	fourierCutoff = int(np.sqrt(-8 / xi**2 * np.log(tol)) / (2 * np.pi))
	
	field = np.zeros(_pos.shape)
	kn = 2 * np.pi * np.arange(-fourierCutoff, fourierCutoff+1)
	km = 2 * np.pi * np.arange(-fourierCutoff, fourierCutoff+1)

	kn, km = meshgrid(kn, km)
	prefactor =  np.exp(-(kn**2 + km**2) * xi**2 / 8)

	for d in range(dims):
		A = np.random.normal(0, 1/np.sqrt(2*np.pi), size=(2*fourierCutoff+1, 2*fourierCutoff+1))
		B = np.random.uniform(0, 2 * np.pi, size=(2*fourierCutoff+1, 2*fourierCutoff+1))
		d_prefactor = A * prefactor
        
		for pID in range(num_particles):
			field[pID, d] = np.sum(d_prefactor * np.cos(B + kn * _pos[pID, 0] + km * _pos[pID, 1]))
		
	return field / L * np.sqrt(2 * sigma * np.pi / (xi * L))

def iterateTimeStep(pos, xi, sigma, tol, D):
	"""Iterate system forward one time step

	Parameters
	----------
	pos : numpy array
		Particle positions. Should have have shape (nParticles, 2)
	xi : float
		Correlation length of the field
	sigma : float
		Disorder parameter
	tol : float
		Tolerance of field generation. Smaller tol means more Fourier 
		modes are included. Should probably be < 0.1
	"""

	field = generateGCF2D(pos, xi, sigma, tol)
	pos += np.random.normal(field, np.sqrt(2 * D))
	return pos

def evolveAndSave(tMax, N, xi, sigma, tol, D, save_file):
	f = open(save_file, "a")
	writer = csv.writer(f)
	writer.writerow(['Time', 'Position'])
	
	times = np.unique(np.geomspace(1, tMax, 500).astype(int))

	positions = np.zeros((N, 2))
	positions += np.random.normal(0, np.sqrt(2 * D), positions.shape)
	for t in range(max(times)):
		positions = iterateTimeStep(positions, xi, sigma, tol, D)
		if t in times:
			writer.writerow([t+1, max(np.sqrt(positions[:, 0]**2 + positions[:, 1]**2 ))])
			f.flush()

	f.close()