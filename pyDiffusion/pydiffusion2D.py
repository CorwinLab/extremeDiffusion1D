import numpy as np
from numba import njit
import csv

@njit
def generateGCF(pos, xi, tol=0.01):
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

@njit 
def iterateTimeStep(positions, xi):
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

	maxPos : numpy array 
		(x, y) coordinates of particle furthest from the origin

	Examples
	--------
	nParticles = 1000
	positions = np.zeros(shape=(nParticles, 2))
	positions, maxPos = iterateTimeStep(positions, 1)
	fig, ax = plt.subplots()
	ax.scatter(positions[:, 0], positions[:, 1])
	ax.scatter(maxPos[0], maxPos[1], c='r', marker='^')
	fig.savefig("Time0.pdf", bbox_inches='tight')
	positions, maxPos = iterateTimeStep(positions, 1)
	fig, ax = plt.subplots()
	ax.scatter(positions[:, 0], positions[:, 1])
	ax.scatter(maxPos[0], maxPos[1], c='r', marker='^')
	fig.savefig("Time1.pdf", bbox_inches='tight')
	'''
	biases = generateGCF(positions, xi)
	num_particles = positions.shape[0]
	for idx in range(num_particles):
		dx = [np.random.normal(biases[idx, 0], xi), np.random.normal(biases[idx, 1], xi)]
		positions[idx, :] += np.array(dx)
	maxIdx = np.argmax(positions[:, 0]**2 + positions[:, 1]**2)
	maxPos = positions[maxIdx, :]
	return positions, maxPos

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
		if t in save_times:
			writer.writerow([t, np.max(positions)])
			f.flush()

	f.close()
	np.savetxt(save_positions, positions)