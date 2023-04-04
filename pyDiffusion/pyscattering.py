import numpy as np
from numba import njit, jit
import csv

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

	for i in range(1, right.size - 1):
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

def evolveAndGetQuantile(times, N, size, beta, save_file):
	right = np.zeros(size+1)
	left = np.zeros(size+1)

	# Start with all the particles moving to the right
	right[right.size // 2] = 1

	f = open(save_file, 'a')
	writer = csv.writer(f)
	writer.writerow(['Time', 'Position'])
	f.flush()

	for t in range(max(times)):
		right_new, left_new, pos = iteratePDF(right[size // 2 - t - 2: size // 2 + t + 3], left[size // 2 - t - 2: size // 2 + t + 3], 1/N, beta=beta)
		right[size // 2 - t - 2: size // 2 + t + 3] = right_new 
		left[size // 2 - t - 2: size // 2 + t + 3] = left_new
		assert np.abs(np.sum(right + left)-1) < 1e-10, np.abs(np.sum(right + left)-1)

		if t in times:
			writer.writerow([t+1, pos])
			f.flush()
	f.close()
	
@jit
def biasingField(xvals, correlation_length):
	grid = np.arange(np.min(xvals) - 3 * correlation_length, np.max(xvals) + 3 * correlation_length, step=1)
	noise = np.random.uniform(0, 1, len(grid))

	kernel_x = np.arange(-3 * correlation_length, 3 * correlation_length, 1)
	kernel = np.exp(-kernel_x**2 / correlation_length**2)
	field = np.convolve(kernel, noise, 'same')
	
	assert np.all(np.diff(grid) > 0), "Sampling points on grid are not monotonically increasing"
	field = np.interp(xvals, grid, field)
	scaling_factor = np.sqrt(1 / correlation_length  ** 2 / np.pi)
	field *= scaling_factor
	return field

@jit
def iteratePDFFields(right, left, quantile, xvals, rc=2):
	biases = biasingField(xvals, rc)
	right_new = np.zeros(right.shape)
	left_new = np.zeros(left.shape)
	cdf_new = 0
	quantileSet = False

	for i in range(1, right.size - 1):
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
