import numpy as np
from numba import njit, vectorize
import csv
import mpmath
import os
import pandas as pd
import sys
from math import gamma

@njit
def randomUniform(size):
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

# This doesn't work with numba
# def getRandVals():
# 	n = 3
# 	alpha = 2
# 	beta= 4
# 	xvals = np.arange(0, n+1)
# 	rand_vals = np.array([0, *betabinom.pmf(xvals, n, alpha, beta)])
# 	rand_vals[3] -= 3.3306690738754696e-16
# 	flip = np.random.choice([0, 1])
# 	if flip: 
# 		return np.flip(rand_vals)
# 	return rand_vals 

LOOKUP_TABLE = np.array([
	1, 1, 2, 6, 24, 120, 720, 5040, 40320,
	362880, 3628800, 39916800, 479001600,
	6227020800, 87178291200, 1307674368000,
	20922789888000, 355687428096000, 6402373705728000,
	121645100408832000, 2432902008176640000], dtype='int64')

@njit
def factorial(n):
	if n > 20:
		raise ValueError
	return LOOKUP_TABLE[n]

@vectorize 
def binom(n, x):
	return factorial(n) / factorial(x) / factorial(n-x)

@vectorize
def B(x, y):
	return gamma(x) * gamma(y) / gamma(x+y)
@njit 
def betaBinomPMF(x, n, alpha, beta):
	return binom(n, x) * B(x + alpha, n-x+beta) / B(alpha, beta)

@njit
def randBetaBinom():
	n = 3
	alpha = 2
	beta= 4
	xvals = np.arange(0, n+1)
	rand_vals = betaBinomPMF(xvals, n, alpha, beta)	
	rand_vals = np.flip(rand_vals)
	rand_vals = np.append(rand_vals, 0)
	rand_vals[1] -= 8.326672684688674e-17
	flip = np.random.choice(np.array([0, 1]))
	if flip: 
		return np.flip(rand_vals)
	return rand_vals

@njit
def randomFourthMoment():
	'''
	xvals = np.array([-2, -1, 0, 1, 2])
	num_samples = 10000
	second = np.zeros(num_samples)
	for i in range(num_samples):
		vals = getRandVals(5, 'randomFourthMomet')
		print(np.sum(xvals * vals))
		print(np.sum(xvals ** 2 * vals))
		print(np.sum(xvals**3 * vals))
	'''

	mn1 = np.random.uniform(1/6, 5/12)
	mn2 = 5/48 - mn1 /4
	m0 = 7/8 -3/2 * mn1
	m1 = mn1 - 1/6
	m2 = 1/16*(3-4*mn1)
	return np.array([mn2, mn1, m0, m1, m2])

@njit
def randomDelta(size):
	xvals = np.arange(0, size, 1)
	rand_vals = np.random.choice(xvals, size=2, replace=False)
	biases = np.zeros(xvals.size)
	biases[rand_vals] = 1/2
	return biases

@njit
def randomThreeStep():
	eps = np.random.uniform(0, 1e-10)
	rand_vals = np.array([1/3 + eps/2, 1/3 - eps, 1/3 + eps/2])
	return rand_vals

@njit
def symmetricRandomDirichlet(alphas):
	rand_vals = randomDirichlet(alphas)
	return (rand_vals + np.flip(rand_vals)) / 2

@vectorize
def gammaDist(alpha, scale):
	return np.random.gamma(alpha, scale)

@njit 
def randomDirichlet(alphas):
	gammas = gammaDist(alphas, np.ones(alphas.shape))
	return gammas / np.sum(gammas)

@njit 
def randomGauss(size):
	G = np.random.normal(0, 1, size=size)
	rand_vars = np.exp(-G)
	return rand_vars / np.sum(rand_vars)

@njit
def ssrw(size):
	return np.ones(size) / size

@njit
def rwre(size):
	rand_val = np.random.uniform(0, 1)
	biases = np.zeros(size)
	biases[0] = rand_val 
	biases[-1] = 1-rand_val
	return biases

@njit
def randomRightTriangle(size):
	rand_val = np.random.triangular(1/4,1/4,1)
	biases = np.zeros(size)
	biases[0] = rand_val
	biases[-1] = 1-rand_val
	return biases

@njit
def rwreBiased():
	""" 
	num_samples = 1000000
	mean = 0
	xvals = np.arange(-2, 3, 1)
	
	for _ in range(num_samples): 
		rand_vals = rwreBiased()
		mean += np.sum(xvals * rand_vals)
	print(mean / num_samples)
	"""
	rand_val = np.random.uniform(1/3, 1)
	biases = np.zeros(5)
	biases[1] = rand_val 
	biases[-1] = 1 - rand_val
	return biases

@njit
def threeStepUniform():
	"""	
	Examples
	--------
	num_samples = 1000000
	mean = np.zeros(3)

	for _ in range(num_samples):
		rand_val = threeStepUniform()
		mean += rand_val
		assert np.all(rand_val >= 0)

	mean /= num_samples
	print(mean)
	"""

	biases = np.zeros(3)
	biases[0] = 1/4
	rand_val = np.random.uniform(0, 1/2)
	biases[2] = rand_val 
	biases[1] = 3/4 - rand_val 
	return biases

@njit 
def nnssrw():
	return np.array([1/2, 0, 1/2])

@njit
def thirdMoment():
	'''
	Produces a random distribution with a mean of 0 and variance of 2
	'''
	b = np.random.uniform(1/5 - 1/15, 1/5 + 1/15)
	c = np.random.uniform(1/5 - 1/15, 1/5 + 1/15)
	a=1/6*(2-3*b-c)
	d=1/3*(2-3*b-4*c)
	e=(b+c)/2
	vals = np.array([a, b, c, d, e])
	return vals

@njit
def thirdMomentDHalf():
	m1 = np.random.uniform(0, 1/11)
	mn1 = np.random.uniform(0, 1/6)
	m0 = 1/8*(7 - 6*mn1 - 6*m1)
	mn2 = 1/16 * (1- 6*mn1 + 2 * m1)
	m2 = 1/16*(1 + 2 * mn1 - 6*m1)

	return np.array([mn2, mn1, m0, m1, m2])

@njit
def thirdMoment7():
	'''
	Produces a random distribution with mean 0 and variance of 2
	'''
	m1 = np.random.uniform(0, 1/9)
	m2 = np.random.uniform(0, 1/9)
	mn1 = np.random.uniform(0, 1/9)
	mn2 = np.random.uniform(0, 1/9)
	
	mn3 =  1/9 * (1-5*mn2 - 2*mn1 + m1 + m2)
	m0 = 1/9*(7-5*mn2 -8*mn1 - 8*m1 -5*m2)
	m3 = 1/9*(1+mn2 + mn1 - 2*m1 - 5*m2)

	return np.array([mn3, mn2, mn1, m0, m1, m2, m3])

@njit
def sticky():
	rand_vals = np.array([0.1, 0, 0.9])
	flip = np.random.choice(np.array([0, 1]))
	if flip: 
		return np.flip(rand_vals)
	return rand_vals

@njit
def constDiffusionCoefficient(k):
	"""Produces a distribution with mean 0 and diffusion coefficient of 10."""
	sigma2 = 10
	xvals = np.arange(-k, k+1)
	rand_vals = np.zeros(2 * k + 1)
	xval = np.random.randint(4, k+1)
	rand_vals[xvals == xval] = sigma2 / 2 / xval**2
	rand_vals[xvals == -xval] = sigma2 / 2 / xval**2
	rand_vals[xvals == 0] = 1 - sigma2 / xval**2

	return rand_vals

@njit
def getRandVals(step_size, distribution, params=np.array([])):
	if distribution == 'symmetric':
		rand_vals = symmetricRandomDirichlet(params)
	elif distribution == 'uniform': # 'notsymmetric'
		rand_vals = randomUniform(step_size)
	elif distribution == 'ssrw':
		rand_vals=ssrw(step_size)
	elif distribution == 'nnssrw':
		rand_vals = nnssrw()
	elif distribution == 'delta':
		rand_vals = randomDelta(step_size)
	elif distribution == 'rwre':
		rand_vals = rwre(step_size)
	elif distribution == 'dirichlet':
		rand_vals = randomDirichlet(params)
	elif distribution == 'righttriangle':
		rand_vals = randomRightTriangle(step_size)
	elif distribution == 'rwreBiased':
		rand_vals = rwreBiased()
	elif distribution == 'threeStepUniform':
		rand_vals = threeStepUniform()
	elif distribution == 'thirdMoment':
		rand_vals = thirdMoment()
	elif distribution == 'thirdMoment7':
		rand_vals = thirdMoment7()
	elif distribution == 'randomThreeStep':
		rand_vals = randomThreeStep()
	elif distribution == 'betaBinom':
		rand_vals = randBetaBinom()
	elif distribution == 'thirdMomentDHalf':
		rand_vals = thirdMomentDHalf()
	elif distribution == 'randomFourthMoment':
		rand_vals = randomFourthMoment()
	elif distribution == 'constDiffusionCoefficient':
		rand_vals = constDiffusionCoefficient(step_size // 2)
	elif distribution == 'sticky':
		rand_vals = sticky()
	return rand_vals

@njit
def iterateTimeStep(pdf, t, step_size=3, distribution='uniform', params=np.array([])):
	'''
	Examples
	--------
	L = 3
	step_size = 5
	pdf = np.zeros(L**2 + step_size * 2)
	pdf[0] = 1
	t = 1
	for _ in range(5):
		maxIdx = 10
		pdf = iterateTimeStep(pdf, t, step_size, 'notsymmetric')
		print(pdf)
		getMeanVarMax(pdf, 100, t, step_size)
		t += 1 
	'''
	pdf_new = np.zeros(pdf.size)
	
	# If we're using the rwre we can avoid the holes by 
	# incrementing the walk
	if distribution == 'rwre':
		increment = step_size // 2
	else:
		increment = 1
	
	# I'm not entirely sure how/why but using this end point means
	# that we iterate over the entire array but no further
	for i in range(0, t * (step_size-1) - step_size + 2, increment):
		if pdf[i] == 0:
			continue
		rand_vals = getRandVals(step_size, distribution, params)
		pdf_new[i: i + step_size] += rand_vals * pdf[i]

	return pdf_new

def iterateNParticles(occ, t, step_size, distribution='uniform', params=np.array([])):
	occ_new = np.zeros(occ.size)
	
	# If we're using the rwre we can avoid the holes by 
	# incrementing the walk
	if distribution == 'rwre':
		increment = step_size // 2
	else:
		increment = 1
	
	# I'm not entirely sure how/why but using this end point means
	# that we iterate over the entire array but no further
	
	for i in range(0, t * (step_size-1) - step_size + 2, increment):
		if occ[i] == 0:
			continue
		rand_vals = getRandVals(step_size, distribution, params)
		occ_new[i: i + step_size] += np.random.multinomial(int(occ[i]), rand_vals)

	return occ_new

@njit
def iterateFPT(pdf, maxIdx, step_size, distribution='uniform', params=np.array([])):
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
		rand_vals = getRandVals(step_size, distribution, params)
		
		# Iterate through rand_vals and appropriately add to pdf_new
		for j in range(len(rand_vals)):
			if j-i < 0:
				pdf_new[0] += pdf[i] * rand_vals[j]
			else:
				pdf_new[j-i] += pdf[i] * rand_vals[j]
		
	for i in range(width, maxIdx):
		# Generate randomt transition biases
		rand_vals = getRandVals(step_size, distribution, params)
		
		pdf_new[i - width : i + width + 1] += rand_vals * pdf[i]
	
	return pdf_new

def evolveAndMeasureFPT(Lmax, step_size, distribution, save_file, N, params=np.array([])):
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
		# This was previously misnamed Mean(Sam) and Var(Sam)
		writer.writerow(["Distance", "Env", "Mean(Min)", "Var(Min)", "PDF Sum"])

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
			pdf = iterateFPT(pdf, maxIdx, step_size, distribution, params)
			t += 1

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

def evolveAndMeasureFPTNoAbsorbing(Lmax, step_size, distribution, save_file, N, params=np.array([])):
	'''
	Calculate the FPT except with no absorbing boundary condition.

	Example
	-------
	Lmax = 50
	step_size = 5
	distribution='uniform'
	save_file = 'Quantile.txt'
	N = 100
	evolveAndMeasureFPTNoAbsorbing(Lmax, step_size, distribution, save_file, N)
	'''
	
	# Get save distances
	Ls = np.unique(np.geomspace(1, Lmax, 500).astype(int))
	
	# Check if save_file is already written to
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
		writer.writerow(["Distance", "Env", "Mean(Min)", "Var(Min)", "PDF Sum"])
	f.flush()
	
	mpmath.mp.dps = 250
	N = mpmath.mp.mpf(N)
	
	for L in Ls:
		# Initialize PDF
		# Initialize the probability distribution
		size = 2 * int(1e6) # np.max(times) * step_size * 5
		pdf = np.zeros(size)
		pdf[0] = 1
		t = 0

		# Initialize quantile and sampling variables
		quantile = None 
		running_sum_squared = 0
		running_sum = 0
		
		# Set up fpt cdf and N first passage CDF
		firstPassageCDF = mpmath.mp.mpf(0)
		nFirstPassageCDFPrev = 1 - (1-firstPassageCDF)**N

		while (1-nFirstPassageCDFPrev > np.finfo(pdf[0].dtype).eps) or (firstPassageCDF < 1 / N):
			pdf = iterateTimeStep(pdf, t+1, step_size, distribution, params)
			assert np.all(pdf >= 0)
			t+=1

			# Need to parse only part of array that is nonzero
			maxIdx = (t+1) * (step_size-1) - step_size + 2
			cdf = np.cumsum(pdf[:maxIdx+1][::-1] * mpmath.mp.mpf(1))[::-1] # Need to convert to mpmath object
			cdf = np.insert(cdf, 0, 1)
			
			xvals = np.arange(0, cdf.size) - t * (step_size // 2)
			
			if L > xvals[-1]:
				firstPassageCDF = mpmath.mp.mpf(0)
			else:
				idx_of_L = np.where(xvals==L)[0][0]
				firstPassageCDF = cdf[idx_of_L]
			
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
	idx = int(x + center) 
	return pdf[idx], np.sum(pdf[idx:])

def getMeanVarMax(pdf, N, t, step_size):
	'''
	Examples
	--------
	L = 3
	step_size = 5
	pdf = np.zeros(L**2 + step_size * 2)
	pdf[0] = 1
	t = 1
	for _ in range(5):
		maxIdx = 50
		pdf = iterateTimeStep(pdf, t, step_size, 'notsymmetric')
		mean, var = getMeanVarMax(pdf, 100, t, step_size)
		print(mean, var)
		t += 1 
	'''
	mpmath.mp.dps = 250
	# Convert N and cdf to mpmath precision
	N = mpmath.mp.mpf(N)

	# Need to parse only part of array that is nonzero
	maxIdx = (t+1) * (step_size-1) - step_size + 2
	cdf = np.cumsum(pdf[:maxIdx+1]* mpmath.mp.mpf(1)) # Need to convert to mpmath object
	cdf = np.insert(cdf, 0, 0)
	
	N_cdf = cdf**N
	N_cdf = np.insert(N_cdf, 0, 0)

	# Pretty sure these are the correct x-vals
	xvals = np.arange(0, cdf.size) - t * (step_size // 2)
	Npdf = np.diff(N_cdf) 
	
	# I think there are issues with the pdf not being within precision so normalizing
	Npdf /= np.sum(Npdf)
	
	# Calculate Mean and Variance 
	mean = np.sum(xvals * Npdf)
	var = np.sum(xvals**2 * Npdf) - mean**2
	
	# assert var >= 0, var
	return float(mean), float(var), float(np.sum(Npdf))

@njit
def measureQuantile(pdf, N, t, step_size):
	cdf = 0 
	for i in np.arange(pdf.size-1, -1, -1):
		cdf += pdf[i]
		if cdf >= 1/N:
			center = t * (step_size // 2)
			return i - center 

def evolveAndMeasureEnvAndMax(tMax, step_size, N, save_file, distribution='uniform', params=np.array([])):
	# Ensure the step_size is odd 
	assert (step_size % 2) != 0, f"Step size is not an odd number but {step_size}"

	# Get save times
	times = np.unique(np.geomspace(1, tMax, 2500).astype(int))

	# Initialize the probability distribution
	size = 2*int(1e6) # np.max(times) * step_size * 5
	pdf = np.zeros(size)
	pdf[0] = 1
	t = 0
	
	# Check if save_file is already written to
	write_header = True
	if os.path.exists(save_file):
		data = pd.read_csv(save_file)
		max_time = max(data['Time'].values)
		if max_time == max(times):
			print("File already completed", flush=True)
			sys.exit()
		times = times[times > max_time]
		print(f"Starting at t={times[0]}", flush=True)
		write_header = False
		
	# Set up writer and write header if save file doesn't exist
	f = open(save_file, 'a')
	writer = csv.writer(f)
	if write_header:
		writer.writerow(["Time", "Env", "Mean(Max)", "Var(Max)", "PDF", "CDF", "PDF Sum", "Npdfsum"])
	f.flush()
	
	maxTime = np.max(times)
	
	while t < maxTime: 
		# Iterate timestep and check all vals are > 0
		pdf = iterateTimeStep(pdf, t+1, step_size, distribution, params)
		assert np.all(pdf >= 0)
		t+=1
		
		if t in times: 
			# Measure the value of Env
			quantile = measureQuantile(pdf, N, t, step_size)

			# Get x=t^3/4 pdf and cdf value
			pdf_val, cdf_val = measurePDFandCDF(pdf, t**(3/4), t, step_size)
			
			# Get mean and var of Max
			mean, var, NpdfSum = getMeanVarMax(pdf, N, t, step_size)

			# Check the variance is positive and save pdf if not
			if var < 0: 
				maxIdx = (t+1) * (step_size-1) - step_size + 2

				save_pdf = pdf[:maxIdx+1]
				save_dir = os.path.dirname(save_file)

				np.savetxt(os.path.join(save_dir, "CDF.txt"), save_pdf)
				np.savetxt(os.path.join(save_dir, "NegVar.txt"), [t, var])

				raise ValueError(f"Variance {var} < 0")
			
			writer.writerow([t, quantile, mean, var, pdf_val, cdf_val, np.sum(pdf), NpdfSum])
			f.flush()

def evolveAndMeasurePDFSymmetric(tMax, step_size, vs, alpha, save_file, distribution='uniform', params=np.array([])):
	# Ensure the step_size is odd 
	assert (step_size % 2) != 0, f"Step size is not an odd number but {step_size}"

	# Get save times
	times = np.unique(np.geomspace(1, tMax, 2500).astype(int))

	# Initialize the probability distribution
	size = 2*int(1e6) # np.max(times) * step_size * 5
	pdf = np.zeros(size)
	pdf[0] = 1
	t = 0
	
	# Check if save_file is already written to
	write_header = True
	if os.path.exists(save_file):
		data = pd.read_csv(save_file)
		max_time = max(data['Time'].values)
		if max_time == max(times):
			print("File already completed", flush=True)
			sys.exit()
		times = times[times > max_time]
		print(f"Starting at t={times[0]}", flush=True)
		write_header = False
		
	# Set up writer and write header if save file doesn't exist
	f = open(save_file, 'a')
	writer = csv.writer(f)
	if write_header:
		writer.writerow(["Time", *vs])
	f.flush()
	
	maxTime = np.max(times)
	
	while t < maxTime: 
		# Iterate timestep and check all vals are > 0
		pdf = iterateTimeStep(pdf, t+1, step_size, distribution, params)
		assert np.all(pdf >= 0)
		t+=1
		
		if t in times: 
			# Measure the value of Env
			row = [t]
			# Get x=t^3/4 pdf and cdf value
			for v in vs:
				_, cdf_val = measurePDFandCDF(pdf, v * t**(alpha), t, step_size)
				row.append(cdf_val)

			writer.writerow(row)
			f.flush()

def getBeta(step_size):
	num_samples = 100000
	xvals = np.arange(- (step_size//2), step_size//2 + 1)

	running_sum = 0
	sigma = 0
	mean = 0
	omega_ij = 0

	for _ in range(num_samples):
		rand_vals = randomDelta(step_size)
		running_sum += np.sum(rand_vals * xvals)**2
		sigma += np.sum(rand_vals * xvals**2)
		mean += np.sum(rand_vals * xvals)
		omega_ij += rand_vals[0] * rand_vals[1]
	print("Mean:", mean / num_samples)
	return running_sum / num_samples, sigma / num_samples, omega_ij / num_samples
