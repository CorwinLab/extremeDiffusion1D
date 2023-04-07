import numpy as np
from numba import njit, jit
import csv
import os
import sys

@njit
def iteratePDF(right, left, quantile, dist='beta', params=1):
	if dist == 'beta':
		if params==1:
			# This is to only generate random numbers on the odd values
			# which should be populated
			biases = np.zeros(right.size)
			rand_uniform = np.random.uniform(0, 1, right[::2].size)
			biases[::2] = rand_uniform
		elif params==0:
			biases = np.zeros(right.size)
			rand_uniform = np.random.uniform(0, 1, right[::2].size)
			biases[::2] = rand_uniform
			biases = np.array([np.round(i) for i in biases])
		elif params == np.inf: 
			biases = np.ones(right.shape) / 2
		else:
			biases = np.zeros(right.size)
			rand_vals = np.random.beta(params, params, size=right[::2].size)
			biases[::2] = rand_vals

	elif dist == 'delta':
		biases = np.zeros(right.size)
		rand_nums = np.random.choice([0, 1/2, 1], size=right[::2].size, p=[params, 1-2*params, params])

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

def evolveAndGetQuantile(times, N, size, dist, params, save_file):
	right = np.zeros(size+1)
	left = np.zeros(size+1)

	# Start with all the particles moving to the right
	right[right.size // 2] = 1

	write_header = True 
	# Check if save file has already been created and make sure we don't 
	# redo any times we've already done
	if os.path.exists(save_file):
		data = np.loadtxt(save_file, skiprows=1, delimiter=',')
		max_time = data[-1, 0]
		if max_time == max(times):
			sys.exit()
		times = times[times > max_time]
		write_header = False 

	f = open(save_file, 'a')
	writer = csv.writer(f)

	# Ensure that we don't write a header twice
	if write_header:
		writer.writerow(['Time', 'Position'])
		f.flush()

	for t in range(max(times)):
		# Only want to pass the part of the array that is non-zero
		right_new, left_new, pos = iteratePDF(right[size // 2 - t - 2: size // 2 + t + 3], left[size // 2 - t - 2: size // 2 + t + 3], 1/N, dist=dist, params=params)
		right[size // 2 - t - 2: size // 2 + t + 3] = right_new 
		left[size // 2 - t - 2: size // 2 + t + 3] = left_new

		# Ensure that the sum adds to roughly 1
		assert np.abs(np.sum(right + left)-1) < 1e-10, np.abs(np.sum(right + left)-1)

		if t in times:
			writer.writerow([t+1, pos])
			f.flush()
	f.close()

def evolveAndGetProbs(times, N, size, beta, save_file):
	right = np.zeros(size+1)
	left = np.zeros(size+1)

	# Start with all the particles moving to the right
	right[right.size // 2] = 1

	write_header = True 
	# Check if save file has already been created and make sure we don't 
	# redo any times we've already done
	if os.path.exists(save_file):
		data = np.loadtxt(save_file, skiprows=1, delimiter=',')
		max_time = data[-1, 0]
		if max_time == max(times):
			sys.exit()
		times = times[times > max_time]
		write_header = False 

	f = open(save_file, 'a')
	writer = csv.writer(f)

	# Ensure that we don't write a header twice
	if write_header:
		writer.writerow(['Time', 'Position', 'Prob', 'Delta'])
		f.flush()

	for t in range(max(times)):
		# Only want to pass the part of the array that is non-zero
		right_new, left_new, pos = iteratePDF(right[size // 2 - t - 2: size // 2 + t + 3], left[size // 2 - t - 2: size // 2 + t + 3], 1/N, beta=beta)
		right[size // 2 - t - 2: size // 2 + t + 3] = right_new 
		left[size // 2 - t - 2: size // 2 + t + 3] = left_new

		# Ensure that the sum adds to roughly 1
		assert np.abs(np.sum(right + left)-1) < 1e-10, np.abs(np.sum(right + left)-1)
		
		idx = pos + (right.size // 2) 
		prob = (right+left)[idx]
		delta = (right - left)[idx]

		if t in times:
			writer.writerow([t+1, pos, prob, delta])
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
