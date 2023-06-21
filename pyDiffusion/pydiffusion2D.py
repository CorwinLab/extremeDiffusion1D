import numpy as np
from numba import njit
import csv

@njit
def generateGCF2D(pos, xi, tol=0.01):
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
	xi = 2
	xmax = 10
	x = np.arange(0,xmax,.1)
	y = np.arange(0,xmax,.1)
	xx, yy = np.meshgrid(x, y)
	x = xx.flatten()
	y = yy.flatten()
	pos = np.vstack([x, y]).T
	c = generateGCF(pos, xi=xi)
	c = c.astype('float64')
	fig, ax = plt.subplots()
	qp = ax.quiver(pos[:, 0], pos[:, 1], c[:, 0], c[:, 1], np.sqrt(c[:, 0]**2 + c[:, 1]**2), angles='xy', scale=1)
	ax.set_title(f"xi = {xi}")
	fig.colorbar(qp, ax=ax)
	fig.savefig(f"GCF{xi}.pdf", bbox_inches="tight")
	"""
	_pos = pos.copy()
	num_particles, dims = _pos.shape
	
	_pos[:, 0] -= np.min(_pos[:, 0])
	_pos[:, 1] -= np.min(_pos[:, 1])

	Lx = (np.max(_pos[:,0]) - np.min(_pos[:,0])) + 3 * xi
	Ly = (np.max(_pos[:,1]) - np.min(_pos[:,1])) + 3 * xi
	L = np.max(np.array([Lx, Ly]))
	fourierCutoff = int(np.sqrt(-L**2 * 8 * np.log(tol)/(2 * np.pi**2)**2))

	# Coerce all particles to be in the box from [0, 1]
	_pos[:, 0] /= L # need to account for minimum values! (i.e. if the minimum is not 0)
	_pos[:, 1] /= L
	xi /= L
	#theta = np.random.uniform(0, 2*np.pi)

	field = np.zeros(_pos.shape)
	
	for d in range(dims):
		A = np.random.normal(0, 1/np.sqrt(2*np.pi), size=(2*fourierCutoff, 2*fourierCutoff))
		B = np.random.uniform(0, 2 * np.pi, size=(2*fourierCutoff, 2*fourierCutoff))
		for pID in range(num_particles):
			for n in np.arange(-fourierCutoff, fourierCutoff): # I think negative values of n means we're iterating over the same values of A[-n]
				kn = 2 * np.pi * n
				for m in np.arange(-fourierCutoff, fourierCutoff): 
					km = 2 * np.pi * m
					# This could shift the two point correlator to something we don't want 
					#xrot = np.cos(theta) * pos[pID, 0] - np.sin(theta) * pos[pID, 1]
					#yrot = np.sin(theta) * pos[pID, 0] + np.cos(theta) * pos[pID, 1]
					field[pID, d] += np.sqrt(2) * np.pi * A[n+fourierCutoff, m+fourierCutoff] * np.exp(-(kn**2 + km**2) * xi**2 / 8) * np.cos(B[n+fourierCutoff, m+fourierCutoff] + kn * _pos[pID, 0] + km * _pos[pID, 1])

	return field / L

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
			field[pID] +=  A[n+fourierCutoff] * np.exp(-kn**2 * xi**2 / 8) * np.cos(B[n+fourierCutoff] + kn * _pos[pID])

	return field / np.sqrt(L) * 2 * np.sqrt(sigma * np.pi / (xi*L))

def getGCF1D(positions, correlation_length, sigma, grid_spacing=0.1):
	'''
	Generate a gaussian correlated field at given positions and correlation length.  

	Paramters
	---------
	positions : numpy array (1D)
		Positions of particles
	
	correlation_length : float 
		Correlation length in space of the field 
	
	D : float 
		Diffusion coefficient to use 
	
	grid_spacing : float (optional 0.1)
		Grid spacing of the random field to generate. 
	
	Returns
	-------
	field : numpy array
		Field stregnth at each particle position.
	
	Example
	-------
	from matplotlib import pyplot as plt
	x = np.random.normal(0, 100, size=100000)
	correlation_length = 10
	field = getGCF1D(x, correlation_length=correlation_length, D=1, grid_spacing=0.1)
	fig, ax = plt.subplots()
	ax.scatter(x, field)
	ax.set_xlabel("Position")
	ax.set_ylabel("Field Strength")
	ax.set_title(f"Correlation Length = {correlation_length}")
	fig.savefig(f"TestingField{correlation_length}.png", bbox_inches='tight')
	'''
	positions = np.array(positions) / grid_spacing
	correlation_length = correlation_length / grid_spacing

	grid = np.arange(np.min(positions) - 3 * correlation_length, np.max(positions) + 3 * correlation_length, step=1)
	noise = np.random.randn(len(grid))

	kernel_x = np.arange(-3 * correlation_length, 3 * correlation_length, 1)
	kernel = np.exp(-kernel_x**2 / correlation_length**2)
	field = np.convolve(kernel, noise, 'same')
	
	assert np.all(np.diff(grid) > 0), "Sampling points on grid are not monotonically increasing"
	field = np.interp(positions, grid, field)
	scaling_factor = np.sqrt(sigma / correlation_length  ** 3 / np.pi)
	return field * scaling_factor / grid_spacing

def iterateTimeStep1D(positions, xi, D, tol, dt):
	'''
	Parameters
	----------
	positions : numpy array 
		Position of particles. Array should have: rows = particleID and
		cols = components

	xi : float 
		Correlation length

	Returns 
	-------
	positions : numpy array 
		Updated position of particles 
	'''
	biases = generateGCF1D(positions, xi, tol)
	positions += np.random.normal(biases * dt, np.sqrt(2 * D * dt))
	return positions

def forwardEuler(f, grid, dx, dt, D, rc, sigma, minIdx, maxIdx, N, tol=0.01):
	'''
	Iterate continuous probability distribution using Euler method. 

	Parameters
	----------
	f : numpy array
		Probability distribution to evolve in time
	
	grid : numpy array
		Real space positions of the probability distribution
	
	dx : float 
		Spacing in position space of grid 
	
	dt : float 
		Time differential 
	
	D : float
		Diffusion coefficient - I think this should be the renormalized
		diffusion coefficient because we're using discrete time steps
	
	rc : float 
		Correlation space in real space 
	
	sigma : float 
		Disorder parameter of the field. Controls the magnitude of the field
	
	tol : float (optional)
		Controls how accurate the two point correlator of the field is

	Note
	----
	Has absorbing boundaries at the edges of the grid. If you hit the boundaries
	the probability distribution will no longer sum to 1. 

	If the diffusion coefficient is not renormalized corretly, there could be 
	negative probabilities. 
	'''
	
	f_new = np.zeros(f.shape)
	field = generateGCF1D(grid[minIdx:maxIdx+1], rc, sigma, tol)
	field_times_prob = f[minIdx:maxIdx+1] * field
	f_new_sum = 0
	quantile = np.nan
	for i in range(minIdx, maxIdx):
		f_new[i] = (D / dx**2 * (f[i+1] - 2 * f[i] + f[i-1]) - 1/(2*dx) * (field_times_prob[i-minIdx+1] - field_times_prob[i-minIdx-1])) * dt + f[i]
		f_new_sum += f_new[i]
		if (f_new_sum >= 1 - 1/N) and (np.isnan(quantile)):
			quantile = grid[i]
	return f_new, quantile

def evolveAndSaveQuantile(L, dx, dt, D, rc, sigma, maxIters, N, tol=0.01):
	grid = np.arange(-L, L+dx, dx)
	centerIdx = grid.size // 2
	p = np.zeros(grid.shape)
	p[p.size // 2] = 1
	t=0
	quantiles = np.zeros(maxIters)
	for i in range(maxIters):
		minIdx = centerIdx - i - 2
		if minIdx == 0:
			raise ValueError("Cannot iterate past grid length")
		maxIdx = centerIdx + i + 3
		p, quantile = forwardEuler(p, grid, dx, dt, D, rc, sigma, minIdx, maxIdx, N, tol)
		t+=dt
		quantiles[i] = quantile
		print(i, t, quantile)
		if (quantile - quantiles[i-1] < 0):
			print("Quantile Decreased")
	return quantiles, t

def evolveAndSaveMaxDistance1D(nParticles, save_times, xi, D, tol, dt, save_file, save_positions):
	f = open(save_file, 'a')
	writer = csv.writer(f)
	writer.writerow(['Time', 'Position'])
	f.flush()
	
	positions = np.zeros(shape=(nParticles))
	t = 0
	while t < max(save_times): 
		positions = iterateTimeStep1D(positions, xi, D, tol, dt)
		t += dt
		if round(t, 2) in save_times: # need to deal with floating point issues
			writer.writerow([t, np.max(positions)])
			f.flush()

	f.close()
	np.savetxt(save_positions, positions)