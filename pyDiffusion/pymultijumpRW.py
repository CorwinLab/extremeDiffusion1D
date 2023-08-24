import numpy as np
from numba import njit
import csv 

@njit
def randomDirichlet(size):
	randomGamma = np.random.exponential(1, size=size)
	return randomGamma / np.sum(randomGamma)

@njit
def symmetricRandomDirichlet(size):
	rand_vals = randomDirichlet(size//2) / 2
	return np.hstack((rand_vals, np.array([0]), rand_vals))


@njit
def iterateTimeStep(pdf, t, step_size=3, symmetric=False):
	pdf_new = np.zeros(pdf.size)
	
	# I'm not entirely sure how/why but using this end point means
	# that we iterate over the entire array but no further
	for i in range(0, t * (step_size-1) - step_size + 2):
		if symmetric: 
			rand_vals = symmetricRandomDirichlet(step_size)
		else:
			rand_vals = randomDirichlet(step_size)
		pdf_new[i: i + step_size] += rand_vals * pdf[i]

	return pdf_new

@njit
def iterateFPT(pdf, maxIdx, step_size, symmetric=False):
	""" Iterate pdf for first passage time

	Parameters
	----------
	pdf : numpy array 
		Spatial probability distribution p(x,t)
	maxIdx : int
		Maximum index to iterate to
	step_size : int
		Total width of the jumpy transition kernel. Must be odd
	symmetric : bool, optional
		Whether the transition kernel should be symmetric or not, by default False

	Returns
	-------
	numpy array
		pdf at next timestep

	Examples
	--------
	L = 3
	step_size = 3
	pdf = np.zeros(L**2 + step_size * 2)
	pdf[L] = 1
	t = 1
	for _ in range(5):
		maxIdx = L + (step_size//2) * t
		pdf = iterateFPT(pdf, maxIdx, step_size, False)
		t += 1 

	Raises
	------
	Error if stepsize is not odd

	Error if maxIdx is outside array
	"""
	assert (step_size % 2) != 0, "Step size is not an odd number"
	assert (maxIdx + (step_size//2) + 1 < len(pdf))

	pdf_new = np.zeros(pdf.size)
	
	# Deal with the boundary at 0
	pdf_new[0] += pdf[0]
	width = step_size // 2

	# Need to handle cases when in distance of boundary carefully
	for i in range(1, width):
		if symmetric:
			rand_vals = symmetricRandomDirichlet(step_size)
		else:
			rand_vals = randomDirichlet(step_size)
		
		# Iterate through rand_vals and appropriately add to pdf_new
		for j in range(len(rand_vals)):
			if j-i < 0:
				pdf_new[0] += pdf[i] * rand_vals[j]
			else:
				pdf_new[j-i] += pdf[i] * rand_vals[j]
		
	for i in range(width, maxIdx):
		# Generate randomt transition biases
		if symmetric:
			rand_vals = symmetricRandomDirichlet(step_size)
		else: 
			rand_vals = randomDirichlet(step_size)
		pdf_new[i - width : i + width + 1] += rand_vals * pdf[i]
	
	return pdf_new

def evolveAndMeasureFPT(Lmax, step_size, symmetric, save_file, N):
	"""_summary_

	Parameters
	----------
	Lmax : _type_
		_description_
	step_size : _type_
		_description_
	symmetric : _type_
		_description_
	save_file : _type_
		_description_
	N : _type_
		_description_

	Examples
	--------
	Lmax = 50
	step_size = 11
	symmetric=False 
	save_file = 'Quantile.txt'
	N = 100
	evolveAndMeasureFPT(Lmax, step_size, symmetric, save_file, N)
	"""
	# Get save distances
	Ls = np.unique(np.geomspace(1, Lmax, 1000).astype(int))

	# Initialize save file writer
	f = open(save_file, 'a')
	writer = csv.writer(f)
	writer.writerow(["Distance", "FPT Quantile", "PDF Sum"])

	pdf_size = int(1e6)

	for L in Ls:
		# Initialize PDF
		pdf = np.zeros(pdf_size)
		pdf[L] = 1

		# Initialize quantile and time
		t = 0
		quantile = None 

		while quantile is None:
			# Set maximum index to 
			maxIdx = L + (step_size//2) * (t+1)

			# Iterate PDF and then step the time forward
			pdf = iterateFPT(pdf, maxIdx, step_size, symmetric)
			t += 1

			fpt_cdf = pdf[0]

			if fpt_cdf >= 1/N:
				quantile = t 

		writer.writerow([L, t, np.sum(pdf)])
	f.close()

def measurePDFandCDF(pdf, x, t, step_size):
	center = t * (step_size // 2)
	idx = x + center 
	return pdf[idx], np.sum(pdf[idx:])

@njit	
def measureQuantile(pdf, N, t, step_size):
	cdf = 0 
	for i in range(pdf.size):
		cdf += pdf[i]
		if cdf >= 1/N:
			center = t * (step_size // 2)
			return i - center  

def evolveAndMeasureQuantileVelocity(tMax, step_size, N, v, save_file, symmetric):
	# Ensure the step_size is odd 
	assert (step_size % 2) != 0, f"Step size is not an odd number but {step_size}"

	# Get save times
	times = np.unique(np.geomspace(1, tMax, 2500).astype(int))

	# Initialize the probability distribution
	size = np.max(times) * step_size
	pdf = np.zeros(size)
	pdf[0] = 1
	t = 0

	# Initialize save file writer
	f = open(save_file, "a")
	writer = csv.writer(f)
	writer.writerow(["Time", "Quantile", "PDF", "CDF"])
	f.flush()
	
	maxTime = np.max(times)
	while t < maxTime: 
		pdf = iterateTimeStep(pdf, t+1, step_size, symmetric)
		assert np.all(pdf >= 0)

		t+=1

		if t in times: 
			quantile = measureQuantile(pdf, N, t, step_size)
			x = int(v * t**(3/4))
			pdf_val, cdf_val = measurePDFandCDF(pdf, x, t, step_size)
			writer.writerow([t, np.abs(quantile), x, pdf_val, cdf_val, np.sum(pdf)])
			f.flush()