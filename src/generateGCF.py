import numpy as np
from numba import njit

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

if __name__ == "__main__":
	from matplotlib import pyplot as plt

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
