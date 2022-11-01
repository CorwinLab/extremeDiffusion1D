import numpy as np
from numba import njit
import csv

@njit
def generateGCF(pos, xi, fourierCutoff=20):
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

	num_particles, dims = pos.shape
	Lx = 2*(max(pos[:,0]) - min(pos[:,0])) + 3 * xi
	Ly = 2*(max(pos[:,1]) - min(pos[:,1])) + 3 * xi
	# rotate positions
	theta = np.random.uniform(0, 2*np.pi)

	field = np.zeros(pos.shape)
	for d in range(dims):
		A = np.random.normal(0, 1/(2*np.pi), size=(fourierCutoff, fourierCutoff))
		B = np.random.uniform(0, 2*np.pi, size=(fourierCutoff, fourierCutoff))
		for pID in range(num_particles):
			for n in range(fourierCutoff):
				kn = 2 * np.pi * n / Lx 
				for m in range(fourierCutoff):
					km = -2 * np.pi * m / Ly
					xrot = np.cos(theta) * pos[pID, 0] - np.sin(theta) * pos[pID, 1]
					yrot = np.sin(theta) * pos[pID, 0] + np.cos(theta) * pos[pID, 1]
					field[pID,d] += 2*np.pi*np.sqrt(2)*xi*A[n,m]*np.exp(-(kn**2 + km**2)*xi**2 / 8)*np.cos(B[n,m] + kn * xrot + km * yrot)

	field[:, 0] -= np.mean(field[:, 0])
	field[:, 1] -= np.mean(field[:, 1])
	field /= np.sqrt(np.sum(field**2))
	return field

def getGCF1D(positions, correlation_length, D, grid_spacing=0.1):
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

    noise_points = (np.max(positions) - np.min(positions) + 6 * correlation_length) / grid_spacing
    grid = np.linspace(np.min(positions) - 3 * correlation_length, np.max(positions) + 3 * correlation_length, int(noise_points))
    noise = np.random.randn(int(noise_points))
    
    kernel_x = np.arange(-3 * correlation_length, 3 * correlation_length, grid_spacing)
    kernel = np.sqrt(D/correlation_length / np.sqrt(np.pi)) * np.exp(-kernel_x**2/2/correlation_length**2)
    noise = np.convolve(noise, kernel, 'same')
    field = np.interp(positions, grid, noise)
    return field

def iterateTimeStep1D(positions, xi, D):
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
	biases = getGCF1D(positions, xi, D, grid_spacing=0.1)
	positions += np.random.normal(biases, np.sqrt(2*D))
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

def evolveAndSaveMaxDistance1D(nParticles, save_times, xi, D, save_file, save_positions):
	f = open(save_file, 'a')
	writer = csv.writer(f)
	writer.writerow(['Time', 'Position'])
	f.flush()
	positions = np.zeros(shape=(nParticles))
	t = 0 
	while t < max(save_times): 
		positions = iterateTimeStep1D(positions, xi, D)
		t+=1
		if t in save_times:
			writer.writerow([t, np.max(positions)])
			f.flush()

	f.close()
	np.savetxt(save_positions, positions)
	
def evolveAndSaveMaxDistance(nParticles, save_times, xi, save_file, save_positions):
	'''
	Paramters
	---------
	nParticles : int
		Number of particles of the system

	save_times : numpy array
		Times to save max position at

	xi : float 
		Correlation length

	save_file : str
		Path of file to save data to

	save_positions: str
		Path of file to save final positions to
	'''
	f = open(save_file, 'a')
	writer = csv.writer(f)
	writer.writerow(['time', 'x', 'y'])
	
	positions = np.zeros(shape=(nParticles, 2))
	t = 0
	while t < max(save_times):
		positions, maxPos = iterateTimeStep(positions, xi)
		t += 1
		if t in save_times: 
			writer.writerow([t, maxPos[0], maxPos[1]])
	f.close()
	np.savetxt(save_positions, positions)
