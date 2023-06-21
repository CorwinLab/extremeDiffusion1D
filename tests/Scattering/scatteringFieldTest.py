import numpy as np 
from matplotlib import pyplot as plt 
from numba import jit

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

if __name__ == '__main__': 
	size = 1000
	correlation_length = 5

	right = np.zeros(size+1)
	left = np.zeros(size+1)
	xvals = np.arange(-size//2, size//2+1, 1)

	right[right.size // 2] = 1
	N = 1e5
	for t in range(size //2):
		right, left, q = iteratePDFFields(right, left, 1/N, xvals, rc=2)
	print(q)
	fig, ax = plt.subplots()
	ax.set_yscale("log")
	ax.set_ylim([10**-20, 1])
	ax.set_xlim([-200, 200])
	ax.plot(xvals[::2], (right+left)[::2])
	fig.savefig("FieldPDF.pdf", bbox_inches='tight')

