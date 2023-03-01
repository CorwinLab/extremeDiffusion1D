import numpy as np
from matplotlib import pyplot as plt
from numba import njit

@njit
def generateGCF(pos, xi, Lx, Ly, tol=0.01):
	_pos = pos.copy()
	num_particles, dims = _pos.shape
	L = np.max(np.array([Lx, Ly]))
	fourierCutoff = int(np.sqrt(-L**2 * 8 * np.log(tol)/(2 * np.pi**2)**2))
	# Coerce all particles to be in the box from [0, 1]
	_pos[0] /= L
	_pos[1] /= L
	xi /= L
	#theta = np.random.uniform(0, 2*np.pi)

	field = np.zeros(_pos.shape)
	A2 = np.zeros(shape=(fourierCutoff, fourierCutoff))
	avg_cos = 0
	avg_cos_num = 0
	for d in range(dims):
		A = np.random.normal(0, 1/np.sqrt(2*np.pi), size=(fourierCutoff, fourierCutoff))
		A2 += A**2
		B = np.random.uniform(0, 2 * np.pi, size=(fourierCutoff, fourierCutoff))
		for pID in range(num_particles):
			for n in np.arange(-fourierCutoff, fourierCutoff):
				kn = 2 * np.pi * n
				for m in np.arange(-fourierCutoff, fourierCutoff):
					km = 2 * np.pi * m
					# This could shift the two point correlator to something we don't want 
					#xrot = np.cos(theta) * pos[pID, 0] - np.sin(theta) * pos[pID, 1]
					#yrot = np.sin(theta) * pos[pID, 0] + np.cos(theta) * pos[pID, 1]
					field[pID, d] += np.sqrt(2) * np.pi * A[n,m] * np.exp(-(kn**2 + km**2) * xi**2 / 8) * np.cos(B[n,m] + kn * _pos[pID, 0] + km * _pos[pID, 1])
					avg_cos += np.cos(B[n,m] + kn * _pos[pID, 0] + km * _pos[pID, 1]) ** 2
					avg_cos_num += 1

	avg_cos /= avg_cos_num
	A2 /= dims
	return field / L, np.sum(A2) / fourierCutoff**2, avg_cos

if __name__ == '__main__':
	Lx = 10
	Ly = 10
	xi = 3
	pos = np.array([0., 0.]).reshape((1, 2))
	tol = 0.0001
	correlator = 0
	A2_sum = 0
	avg_cos_sum = 0
	numSamples = 1000
	for i in range(numSamples):
		field, A2, avg_cos = generateGCF(pos, xi, Lx, Ly, tol=tol)
		A2_sum += A2
		avg_cos_sum += avg_cos
		correlator += np.sum(field**2)
		print(i)
		
	correlator /= numSamples
	A2_sum /= numSamples
	avg_cos_sum /= numSamples
	print(f'Lx={Lx}, Lx={Ly}, tolerance={tol}')
	print("-----------------")
	print('Measured:', correlator)
	print('Expected:', 1 / xi**2)
	print('Calculated:', 2 / xi**2)
	print(f"A^2: {A2_sum}")
	print(f"Expected: {1/(2*np.pi)}")
	print(f"Avg Cos: {avg_cos_sum}")
	