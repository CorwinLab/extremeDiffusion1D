import numpy as np
from numba import njit
import csv

@njit
def generateGCF1D(pos, xi, sigma, tol=0.01):
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
	L = 50
	xi = 5
	sigma = 5
	step = 0.1
	x = np.arange(-L, L+step, step=step)

	field = generateGCF1D(x, xi, sigma, tol=0.001)
	fig, ax = plt.subplots()
	ax.plot(x, field)
	fig.savefig("1DField.png", bbox_inches='tight')
	"""
	
	_pos = pos.copy()
	_pos = (_pos - np.min(_pos)) # need to account for negative minimum value
	
	L = (np.max(_pos) - np.min(_pos)) + 3 * xi
	
	# Coerce all particles to be in the box from [0, 1]
	_pos = _pos / L
	xi = xi / L
	fourierCutoff = int(np.sqrt(-8 / xi**2 * np.log(tol)) / (2 * np.pi))
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
			field[pID] +=  A[n+fourierCutoff] * np.exp(-kn**2 * xi**2 / 8) * np.cos(B[n+fourierCutoff] + kn * _pos[pID])

	return field / np.sqrt(L) * 2 * np.sqrt(sigma * np.pi / (xi*L))

def iterateTimeStep1D(positions, xi, sigma, tol, D):
	'''
	Parameters
	----------
	positions : numpy array 
		Position of particles. Array should have shape (N, 1) where 
		N is the number of particles

	xi : float 
		Correlation length

	Returns 
	-------
	positions : numpy array 
		Updated position of particles 
	
	Example
	-------
	from matplotlib import pyplot as plt
	N = int(1e6)
	xi = 5
	sigma = 1
	tol = 1e4 
	D = 1

	positions = np.zeros(N)
	positions += np.random.normal(0, np.sqrt(2 * D), positions.shape)
	fig, ax = plt.subplots()
	ax.set_xlabel("x")
	ax.set_ylabel("Particle Density")
	ax.hist(positions, bins=50, density=True)
	fig.savefig(f"./Figures/Positions{0}.png", bbox_inches='tight')
	plt.close(fig)
	for i in range(50):
		positions = iterateTimeStep1D(positions, xi, sigma, tol, D)
	
		fig, ax = plt.subplots(figsize=(6.4,4.8))
		ax.set_xlabel("x")
		ax.set_ylabel("Particle Density")
		ax.hist(positions, bins=50, density=True)
		fig.savefig(f"./Figures/Positions{i+1}.png", dpi=100)
		plt.close(fig)
	'''

	biases = generateGCF1D(positions, xi, sigma, tol)
	positions += np.random.normal(biases, np.sqrt(2 * D))
	return positions

def evolveAndSave(tMax, N, xi, sigma, tol, D, save_file):
	f = open(save_file, "a")
	writer = csv.writer(f)
	writer.writerow(['Time', 'Position'])
	
	times = np.unique(np.geomspace(1, tMax, 500).astype(int))

	positions = np.zeros(N)
	for t in range(max(times)):
		positions = iterateTimeStep1D(positions, xi, sigma, tol, D)
		if t in times:
			writer.writerow([t+1, max(positions)])
			f.flush()

	f.close()