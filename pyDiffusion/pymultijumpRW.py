import numpy as np
from numba import njit
import csv
import mpmath
import os
import pandas as pd
import sys

@njit
def randomDirichlet(size):
	'''
	Examples
	--------
	num_samples = 10000
	running_sum_squared = 0
	running_sum = 0
	step_size = 11

	for _ in range(num_samples):
		rand_vals = randomDirichlet(step_size)
		running_sum_squared += rand_vals[0] ** 2 
		running_sum += rand_vals[0]

	running_sum_squared /= num_samples 
	running_sum /= num_samples
	print(running_sum, 1/step_size)
	print(running_sum_squared - running_sum**2, 1/step_size*(1-1/step_size) / (step_size + 1))
	'''
	randomGamma = np.random.exponential(1, size=size)
	return randomGamma / np.sum(randomGamma)

@njit
def randomDelta(size):
	xvals = np.arange(0, size, 1)
	rand_vals = np.random.choice(xvals, size=2, replace=False)
	biases = np.zeros(xvals.size)
	biases[rand_vals] = 1/2
	return biases

@njit
def symmetricRandomDirichlet(size):
	rand_vals = randomDirichlet(size)
	return (rand_vals + np.flip(rand_vals)) / 2

@njit 
def randomGauss(size):
	G = np.random.normal(0, 1, size=size)
	rand_vars = np.exp(-G)
	return rand_vars / np.sum(rand_vars)

@njit
def ssrw(size):
	return np.ones(size) / size

@njit 
def rwre():
	rand_val = np.random.uniform(0, 1)
	return np.array([rand_val, 0, 1-rand_val])

@njit
def getRandVals(step_size, distribution):
	if distribution == 'symmetric':
		rand_vals = symmetricRandomDirichlet(step_size)
	elif distribution == 'notsymmetric':
		rand_vals = randomDirichlet(step_size)
	elif distribution == 'ssrw':
		rand_vals=ssrw(step_size)
	elif distribution == 'delta':
		rand_vals = randomDelta(step_size)
	elif distribution == 'rwre':
		rand_vals = rwre()
	return rand_vals

@njit
def iterateTimeStep(pdf, t, step_size=3, distribution='symmetric'):
	pdf_new = np.zeros(pdf.size)
	
	# I'm not entirely sure how/why but using this end point means
	# that we iterate over the entire array but no further
	for i in range(0, t * (step_size-1) - step_size + 2):
		rand_vals = getRandVals(step_size, distribution)
		pdf_new[i: i + step_size] += rand_vals * pdf[i]

	return pdf_new

@njit
def iterateFPT(pdf, maxIdx, step_size, distribution='symmetric'):
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
		if distribution == 'symmetric':
			rand_vals = symmetricRandomDirichlet(step_size)
		elif distribution == 'notsymmetric':
			rand_vals = randomDirichlet(step_size)
		elif distribution == 'ssrw':
			rand_vals=ssrw(step_size)
		elif distribution == 'delta':
			rand_vals = randomDelta(step_size)
		elif distribution == 'rwre':
			rand_vals = rwre()
		
		# Iterate through rand_vals and appropriately add to pdf_new
		for j in range(len(rand_vals)):
			if j-i < 0:
				pdf_new[0] += pdf[i] * rand_vals[j]
			else:
				pdf_new[j-i] += pdf[i] * rand_vals[j]
		
	for i in range(width, maxIdx):
		# Generate randomt transition biases
		if distribution == 'symmetric':
			rand_vals = symmetricRandomDirichlet(step_size)
		elif distribution == 'notsymmetric':
			rand_vals = randomDirichlet(step_size)
		elif distribution == 'ssrw':
			rand_vals = ssrw(step_size)
		elif distribution == 'delta':
			rand_vals = randomDelta(step_size)
		elif distribution == 'rwre':
			rand_vals = rwre()
		
		pdf_new[i - width : i + width + 1] += rand_vals * pdf[i]
	
	return pdf_new

def evolveAndMeasureFPT(Lmax, step_size, distribution, save_file, N):
	""" Given a maximum position calculate environmental location and 
	sampling mean/variance for the environment.

	Parameters
	----------
	Lmax : _type_
		_description_
	step_size : _type_
		_description_
	distribution : str
		Should be one of ['symmetric', 'notsymmetric', 'ssrw', 'delta']
	save_file : _type_
		_description_
	N : _type_
		_description_

	Examples
	--------
	Lmax = 50
	step_size = 11
	distribution='notsymmetric'
	save_file = 'Quantile.txt'
	N = 100
	evolveAndMeasureFPT(Lmax, step_size, distribution, save_file, N)
	"""
	# Get save distances
	Ls = np.unique(np.geomspace(1, Lmax, 500).astype(int))

	# Check if save file has already been written to
	write_header = True
	if os.path.exists(save_file):
		data = pd.read_csv(save_file)
		max_position = max(data['Distance'].values)
		if max_position == max(Ls):
			print("File already completed", flush=True)
			sys.exit()
		Ls = Ls[Ls > max_position]
		print(f"Starting at {Ls[0]}", flush=True)
		write_header = False

	# Set up writer and write header if save file doesn't exist
	f = open(save_file, 'a')
	writer = csv.writer(f)
	if write_header:
		writer.writerow(["Distance", "Env", "Mean(Sam)", "Var(Sam)", "PDF Sum"])

	pdf_size = int(1e6)
	mpmath.mp.dps = 250
	N = mpmath.mp.mpf(N)

	for L in Ls:
		# Initialize PDF
		pdf = np.zeros(pdf_size)
		pdf[L] = 1

		# Initialize exp variables
		t = 0

		# Initialize quantile and sampling variables
		quantile = None 
		running_sum_squared = 0
		running_sum = 0
		
		# Set up fpt cdf and N first passage CDF
		firstPassageCDF = mpmath.mp.mpf(pdf[0])
		nFirstPassageCDFPrev = 1 - (1-firstPassageCDF)**N

		while (1-nFirstPassageCDFPrev > np.finfo(pdf[0].dtype).eps) or (firstPassageCDF < 1 / N):
			# Set maximum index to 
			maxIdx = L + (step_size//2) * (t+1)

			# Iterate PDF and then step the time forward
			pdf = iterateFPT(pdf, maxIdx, step_size, distribution)
			t += 1
			print(pdf, np.sum(pdf))

			firstPassageCDF = mpmath.mp.mpf(pdf[0])
			nFirstPassageCDF = 1 - (1-firstPassageCDF)**N
			nFirstPassagePDF = nFirstPassageCDF - nFirstPassageCDFPrev
			nFirstPassagePDF = float(nFirstPassagePDF)

			running_sum_squared += t ** 2 * nFirstPassagePDF
			running_sum += t * nFirstPassagePDF
			
			if (quantile is None) and (firstPassageCDF > 1 / N):
				quantile = t
			
			nFirstPassageCDFPrev = nFirstPassageCDF

		variance = running_sum_squared - running_sum ** 2
		writer.writerow([L, quantile, running_sum, variance, np.sum(pdf)])
		f.flush()

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

def evolveAndMeasureQuantileVelocity(tMax, step_size, N, v, save_file, distribution='symmetric'):
	# Ensure the step_size is odd 
	assert (step_size % 2) != 0, f"Step size is not an odd number but {step_size}"

	# Get save times
	times = np.unique(np.geomspace(1, tMax, 2500).astype(int))

	# Initialize the probability distribution
	size = np.max(times) * step_size * 5
	pdf = np.zeros(size)
	pdf[0] = 1
	t = 0

	# Initialize save file writer
	f = open(save_file, "a")
	writer = csv.writer(f)
	writer.writerow(["Time", "Quantile", "x", "PDF", "CDF", "Check Sum"])
	f.flush()
	
	maxTime = np.max(times)
	while t < maxTime: 
		pdf = iterateTimeStep(pdf, t+1, step_size, distribution)
		assert np.all(pdf >= 0)

		t+=1

		if t in times: 
			quantile = measureQuantile(pdf, N, t, step_size)
			x = int(v * t**(3/4))
			pdf_val, cdf_val = measurePDFandCDF(pdf, x, t, step_size)
			writer.writerow([t, np.abs(quantile), x, pdf_val, cdf_val, np.sum(pdf)])
			f.flush()

def getBeta(step_size):
	num_samples = 100000
	xvals = np.arange(- (step_size//2), step_size//2 + 1)

	running_sum = 0
	for _ in range(num_samples):
		rand_vals = randomDelta(step_size)
		running_sum += np.sum(rand_vals * xvals)**2

	return running_sum / num_samples
