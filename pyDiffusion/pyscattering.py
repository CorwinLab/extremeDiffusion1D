import numpy as np
from numba import njit

@njit
def iteratePDF(right, left, quantile):
	biases = np.random.uniform(0, 1, right.size)
	right_new = np.zeros(right.shape)
	left_new = np.zeros(left.shape)
	cdf_new = 0
	quantileSet = False

	for i in range(1, len(right) - 1):
		right_new[i] = right[i-1] * biases[i-1] + left[i-1] * (1-biases[i-1]) 
		left_new[i] = left[i+1] * biases[i+1] + right[i+1] * (1-biases[i+1])
		cdf_new += right_new[i] + left_new[i]
		if (1-cdf_new <= quantile) and not quantileSet: 
			pos = i - (len(right_new) // 2)
			quantileSet = True
	
	return right_new, left_new, pos

def evolveAndGetQuantile(times, N, size):
	right = np.zeros(size+1)
	left = np.zeros(size+1)

	rand_initial = np.random.uniform(0, 1)
	right[right.size // 2] = rand_initial
	left[left.size // 2] = 1-rand_initial

	quantiles = np.zeros(max(times))

	for t in range(max(times)):
		right, left, pos = iteratePDF(right, left, 1/N)
		quantiles[t] = pos 
	
	return quantiles