import numpy as np
from numba import njit

@njit
def iteratePDF(right, left, quantile, beta=1):
	if beta==1:
		# This is to only generate random numbers on the odd values
		# which should be populated
		biases = np.zeros(right.size)
		rand_uniform = np.random.uniform(0, 1, right[::2].size)
		biases[::2] = rand_uniform
	elif beta==0:
		biases = np.zeros(right.size)
		rand_uniform = np.random.uniform(0, 1, right[::2].size)
		biases[::2] = rand_uniform
		biases = np.array([np.round(i) for i in biases])
	elif beta == np.inf: 
		biases = np.ones(right.shape) / 2
	else:
		biases = np.zeros(right.size)
		rand_vals = np.random.beta(beta, beta, size=right[::2].size)
		biases[::2] = rand_vals

	right_new = np.zeros(right.shape)
	left_new = np.zeros(left.shape)
	cdf_new = 0
	quantileSet = False

	for i in range(1, right.size-1):
		# Scattering Model for diffusion
		right_new[i] = right[i-1] * biases[i-1] + left[i-1] * (1-biases[i-1])
		left_new[i] = left[i+1] * biases[i+1] + right[i+1] * (1-biases[i+1])
		
		# RWRE regular diffusion
		#right_new[i] = right[i-1] * biases[i-1] + left[i-1] * (biases[i-1])
		#left_new[i] = left[i+1] * (1-biases[i+1]) + right[i+1] * (1-biases[i+1])

		cdf_new += right_new[i] + left_new[i]
		if (1-cdf_new <= quantile) and not quantileSet: 
			pos = i - (right_new.size // 2)
			quantileSet = True

	return right_new, left_new, pos

if __name__ == '__main__': 
	import time 
	size = 100000
	right = np.zeros(size+1)
	left = np.zeros(size+1)
	right[right.size // 2] = 1
	N = 1e5
	# compile function first 
	iteratePDF(right, left, 1/10, 0)

	right = np.zeros(size+1)
	left = np.zeros(size+1)
	right[right.size // 2] = 1

	right = np.zeros(size+1)
	left = np.zeros(size+1)
	right[right.size // 2] = 1

	start = time.time()
	for t in range(size // 2 - 2):
		right_new, left_new, pos = iteratePDF(right[size // 2 - t - 2: size // 2 + t + 3], left[size // 2 - t - 2: size // 2 + t + 3], 1/N, beta=0.1)
		right[size // 2 - t - 2: size // 2 + t + 3] = right_new 
		left[size // 2 - t - 2: size // 2 + t + 3] = left_new
		assert np.abs(np.sum(right + left)-1) < 1e-10, np.abs(np.sum(right + left)-1)
		print(t)
	print(time.time() - start)