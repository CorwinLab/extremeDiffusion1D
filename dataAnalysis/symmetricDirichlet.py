import numpy as np
from scipy.sparse import linalg
from pyDiffusion.pymultijumpRW import betaBinomPMF

def getSecondMomentArraydirichlet(alpha):
	alpha_0 = np.sum(alpha)
	
	firstMoments = alpha / alpha_0
	
	firstMomentArray = np.zeros(shape = (len(firstMoments), len(firstMoments)))
	secondMomentArray = np.zeros(shape = (len(firstMoments), len(firstMoments)))
	
	# The i, j element will be the covariance of X_i, X_j
	for i in range(secondMomentArray.shape[0]):
		for j in range(secondMomentArray.shape[1]):			
			alpha_i = alpha[i] / alpha_0 
			alpha_j = alpha[j] / alpha_0

			if i != j: 
				secondMomentArray[i, j] += -alpha_i * alpha_j / (alpha_0 + 1) + alpha_i * alpha_j
			else: 
				secondMomentArray[i, j] += alpha_i * (1 - alpha_i) / (alpha_0 + 1) + alpha_i**2
			firstMomentArray[i, j] = firstMoments[i] * firstMoments[j]

	return firstMomentArray, secondMomentArray, firstMoments

def getSecondMomentArraysymmetric(alpha):
	alpha = np.array(alpha)
	k = len(alpha) // 2

	firstMomentArray, secondMomentArray, firstMoments = getSecondMomentArraydirichlet(alpha)
	symmetricArray = np.zeros(shape = (len(alpha), len(alpha)))
	
	firstMomentArray = np.zeros(shape = (len(alpha), len(alpha)))
	firstMoments = alpha / np.sum(alpha)
	firstMoments = (firstMoments + np.flip(firstMoments)) / 2

	for i in range(len(alpha)):
		for j in range(len(alpha)):
			xval = i - k
			yval = j - k
			
			# Need to get negative x, y indeces
			i_n = -xval + k 
			j_n = -yval + k
	
			symmetricArray[i, j] = (secondMomentArray[i, j] + secondMomentArray[i_n, j] + secondMomentArray[i, j_n] + secondMomentArray[i_n, j_n]) / 4
			firstMomentArray[i, j] = firstMoments[i] * firstMoments[j]
	
	return firstMomentArray, symmetricArray, firstMoments

def getSecondMomentArraybetaBinom():
	n = 3
	alpha = 2
	beta= 4
	xvals = np.arange(0, n+1)

	pdf = betaBinomPMF(xvals, n, alpha, beta)	
	pdf = np.flip(pdf)
	pdf = np.append(pdf, 0)
	pdf[1] -= 8.326672684688674e-17
	
	pdf_r = np.flip(pdf)

	# E[w]
	firstMoments = (pdf + pdf_r) / 2

	# E[w(i)] * E[w(j)]
	firstMomentArray = np.zeros(shape = (len(pdf), len(pdf)))

	# E[w(i) w(j)]
	secondMomentArray = np.zeros(shape = (len(pdf), len(pdf)))

	for i in range(len(pdf)):
		for j in range(len(pdf)):
			firstMomentArray[i, j] = firstMoments[i] * firstMoments[j]
			secondMomentArray[i, j] = ((pdf[i] * pdf[j]) + (pdf_r[i] * pdf_r[j])) / 2

	return firstMomentArray, secondMomentArray, firstMoments

def getSecondMomentArraythirdMoment():
	a = 2/15
	b = 4/15

	# E[X2^2] and E[X3^2]
	meanX2X2 = (a**2 + a * b + b**2) / 3

	# E[Xi] - Note: these do indeed add to 1
	meanX3 = meanX2 = (b + a) / 2
	meanX1 = 1/6 * (2 - 3*meanX2 - meanX3)
	meanX4 = 1/3 * (2 - 3 * meanX2 - 4*meanX3)
	meanX5 = (meanX2 + meanX3) / 2

	firstMoments = np.array([meanX1, meanX2, meanX3, meanX4, meanX5])

	firstMomentArray = np.zeros(shape = (len(firstMoments), len(firstMoments)))
	secondMomentArray = np.zeros(shape = (len(firstMoments), len(firstMoments)))
	
	# E[X1^2]
	secondMomentArray[0, 0] = 1/36 * (10 * meanX2X2 + 6 * meanX2 * meanX3 - 16 * meanX2 + 4)
	
	# E[X1 X2] 
	secondMomentArray[0, 1] = 1/6 * (2 * meanX2 - 3 * meanX2X2 - meanX3 * meanX2)

	# E[X1 X3]
	secondMomentArray[0, 2] = 1/6 * (2 * meanX3 - 3 * meanX2 * meanX3 - meanX2X2)

	# E[X1 X4] 
	secondMomentArray[0, 3] = 1/2 * meanX2X2 + 5/6 * meanX2 * meanX3 -2/3 * meanX2 + 2/9 * meanX2X2 -5/9 * meanX3 + 2/9

	# E[X1 X5]
	secondMomentArray[0, 4] = -1/4 * meanX2X2 - 1/3 * meanX2 * meanX3 + 1/6 * meanX2 - 1/12 * meanX2X2 + 1/6 * meanX2 

	# E[X2 X1]
	secondMomentArray[1, 0] = secondMomentArray[0, 1]

	# E[X2 X2]
	secondMomentArray[1, 1] = meanX2X2 

	# E[X2 X3] 
	secondMomentArray[1, 2] = meanX2 * meanX3 

	# E[X2 X4] 
	secondMomentArray[1, 3] = 1/3 * (2 * meanX2 - 3 * meanX2X2 - 4 * meanX2 * meanX3)

	# E[X2 X5]
	secondMomentArray[1, 4] = 1/2 * (meanX2X2 + meanX2 * meanX3)

	# E[X3 X1]
	secondMomentArray[2, 0] =  secondMomentArray[0, 2]

	# E[X3 X2]
	secondMomentArray[2, 1] = secondMomentArray[1, 2]

	# E[X3 X3]
	secondMomentArray[2, 2] = secondMomentArray[1, 1]

	# E[X3 X4]
	secondMomentArray[2, 3] = 1/3 * (2 * meanX2 - 3*meanX2 * meanX3 - 4 * meanX2X2)

	# E[X3 X5] 
	secondMomentArray[2, 4] = secondMomentArray[1, 4] # might not be right

	# E[X4 X1]
	secondMomentArray[3, 0] = secondMomentArray[0, 3]

	# E[X4 X2]
	secondMomentArray[3, 1] = secondMomentArray[1, 3]

	# E[X4 X3] 
	secondMomentArray[3, 2] = secondMomentArray[2, 3]

	# E[X4 X4]
	secondMomentArray[3, 3] = meanX2X2 + 8/3 * meanX2 * meanX2 - 4/3 * meanX2 + 16/9 * meanX2X2 - 16/9 * meanX2 + 4/9

	# E[X4 X5]
	secondMomentArray[3, 4] = -1/2 * meanX2X2 -7/6 * meanX2 * meanX2 + meanX2 / 3 - 2/3 * meanX2X2 + 1/3 * meanX2 

	# E[X5 X1]
	secondMomentArray[4, 0] = secondMomentArray[0, 4]

	# E[X5 X2]
	secondMomentArray[4, 1] = secondMomentArray[1, 4]

	# E[X5 X3]
	secondMomentArray[4, 2] = secondMomentArray[2, 4]

	# E[X5 X4]
	secondMomentArray[4, 3] = secondMomentArray[3, 4]

	# E[X5 X5]
	secondMomentArray[4, 4] = 1/2 * meanX2X2 + 1/2 * meanX2 * meanX3

	for i in range(firstMomentArray.shape[0]):
		for j in range(firstMomentArray.shape[1]):
			firstMomentArray[i, j] = firstMoments[i] * firstMoments[j]

	return firstMomentArray, secondMomentArray, firstMoments

def getSecondMomentArrayrandomFourthMoment():
	firstMoments = np.array([1/32, 7/24, 7/16, 1/8, 11/96])

	secondMomentArray = np.array([	[1/768, 1/128,   1/64,   1/384, 1/256],
									[1/128, 13/144,  23/192, 1/24,  37/1152],
									[1/64,  23/192,  13/64,  3/64,  5/96],
									[1/384, 1/24,    3/64,   1/48,  5/384],
									[1/256, 37/1152, 5/96,   5/384, 31/2304]])

	firstMomentArray = np.zeros(shape = (len(firstMoments), len(firstMoments)))
	for i in range(firstMomentArray.shape[0]):
		for j in range(firstMomentArray.shape[1]):
			firstMomentArray[i, j] = firstMoments[i] * firstMoments[j]

	return firstMomentArray, secondMomentArray, firstMoments

def getSecondMomentArrayconstDiffusionCoefficient(k):
	size = 2 * k + 1
	secondMomentArray = np.zeros(shape = (size, size))
	firstMomentArray = np.zeros(shape = (size, size))

	rand_vals = np.arange(4, k+1)
	dists = []
	sigma2 = 10
	xvals = np.arange(-k, k+1)
	for x in rand_vals:
		rand_vals = np.zeros(size)
		rand_vals[xvals == x] = sigma2 / 2 / x**2
		rand_vals[xvals == -x] = sigma2 / 2 / x**2
		rand_vals[xvals == 0] = 1 - sigma2 / x**2
		dists.append(rand_vals)
	
	dists = np.array(dists)
	
	# E[w]
	firstMoments = np.mean(dists, axis=0)

	for i in range(size):
		for j in range(size):
			firstMomentArray[i, j] = firstMoments[i] * firstMoments[j]

			for dist in dists:
				secondMomentArray[i, j] += dist[i] * dist[j]
			secondMomentArray[i, j] /= dists.shape[0]

	return firstMomentArray, secondMomentArray, firstMoments

def getInvMeasure(size, secondMoments, firstMoments):
	"""Calculate Invariant measure for symmetric alphas"""
	# firstMoments, secondMoments = getSecondMomentArraySymmetricArbitraryAlpha(alphas)
	
	transitionMatrix = np.zeros([size, size])
	k = transitionMatrix.shape[0] // 2
	for i in range(transitionMatrix.shape[0]):
		for j in range(transitionMatrix.shape[1]):
			xval = i - k
			yval = j - k
			if i == transitionMatrix.shape[0] // 2:
				transitionMatrix[i, j] = np.trace(secondMoments, offset= yval)
			else:
				# This might need to change if the mean of the distribution isn't symmetric
				transitionMatrix[i, j] = np.trace(firstMoments, offset= xval - yval) 

	# I think need to take the transpose in order 
	# to get the correct inv measure
	transitionMatrix = transitionMatrix.T
	eigenvalues, eigenvectors = linalg.eigs(transitionMatrix, k=1, which='LM')
	argmax = np.argmax(eigenvalues)
	mu = eigenvectors[:,argmax]
	mu = mu / mu[np.argmax(np.abs(mu))]
	mu = mu[len(mu) // 2: ]
	return mu

def calculateLocalTimeSum(k, secondMomentArray, firstMomentArray, size=501):
	# Calculate first sum which is over |i-j|
	firstSum = 0
	for idx_i in range(secondMomentArray.shape[0]):
		for idx_j in range(secondMomentArray.shape[1]):
			# Convert index in array to real values
			xval = idx_i - (2 * k + 1) // 2
			yval = idx_j - (2 * k + 1) // 2 
			
			firstSum += np.abs(xval - yval) * secondMomentArray[idx_i, idx_j]

	# Calculate second sum which is over (|i-j| - l) * invMeasure
	secondSum = 0
	measure = getInvMeasure(size, secondMomentArray, firstMomentArray)
	for l in range(1, 2 * k + 1):
		# Calculate the sum over |i-j| - l
		sumOverij = 0 
		for idx_i in range(secondMomentArray.shape[0]):
			for idx_j in range(secondMomentArray.shape[1]):
				# Convert index in array to real values
				xval = idx_i - (2 * k + 1) // 2
				yval = idx_j - (2 * k + 1) // 2

				# Add values to sum over i and j
				if np.abs(xval - yval) > l:
					sumOverij += (np.abs(xval - yval) - l) * firstMomentArray[idx_i, idx_j]

		secondSum += sumOverij * 2 * np.real_if_close(measure[l])

	coeff = firstSum + secondSum
	return coeff

def calculateCoefficient(distribution, m, *args):
	# Can calculate the uniform, delta and dirichlet distribution analytically so do that here
	match distribution:
		# I make some really big assumptions about the order of args here 
		case 'uniform':
			width = int(args[0]) // 2
			Dext = width / 12
			D = 1 / 6 * width * (width + 1)
			coeff = Dext / 2 / (D - Dext)
			return coeff, 2 * Dext, D
		case 'delta':
			width = int(args[0]) // 2
			Dext = (2 * width - 1) * (width + 1) / 24
			D = 1/6 * width * (width + 1)
			coeff = Dext / 2 / (D - Dext)
			return coeff, 2 * Dext, D 
		
		case 'dirichlet':
			alpha = np.array(args[0])
			xvals = np.arange(-(len(alpha) // 2), (len(alpha)//2) + 1, 1)
			firstMomentArray, secondMomentArray, firstMoments = getSecondMomentArraydirichlet(alpha)
			Dext = np.sum((secondMomentArray * xvals).T * xvals) / 2
			D  = np.sum(firstMoments * xvals**2) / 2
			coeff = Dext / 2 / (D - Dext)
			return coeff, 2 * Dext, D	
		
	function = f'getSecondMomentArray{distribution}{args}'
	firstMomentArray, secondMomentArray, meanArray = eval(function)
	k = firstMomentArray.shape[0] // 2
	localTimeCoeff = calculateLocalTimeSum(k, secondMomentArray, firstMomentArray)

	var = 0
	for idx_i in range(secondMomentArray.shape[0]):
		for idx_j in range(secondMomentArray.shape[1]):
			xval = idx_i - (2 * k + 1) // 2
			yval = idx_j - (2 * k + 1) // 2

			var += xval**m * yval**m * secondMomentArray[idx_i, idx_j]

	xvals = np.arange(-k, k+1)
	D = 1/2 * np.sum(meanArray * xvals**2)
	avgVar = np.sum(meanArray * xvals**m)
	var -= avgVar ** 2
	coeff = var / localTimeCoeff

	return coeff, var, D
