import numpy as np
from scipy.sparse import linalg
from pyDiffusion.pymultijumpRW import betaBinomPMF

def getSecondMomentArrayUniformRandom():
	arr1 = np.array([0, 1/3, 1/3, 1/3, 0])
	arr2 = np.array([1/5, 1/5, 1/5, 1/5, 1/5])
	secondMomentArray = np.zeros(shape = (len(arr1), len(arr1)))
	for i in range(secondMomentArray.shape[0]):
		for j in range(secondMomentArray.shape[1]):
			secondMomentArray[i,j] = 1/2 * (arr1[i] * arr1[j] + arr2[i] * arr2[j])

	return secondMomentArray

def getSecondMomentArraySymmetricDirichlet(k):
	# I think this is for all alpha = 1
	# Calculate Array which contains Second Moments
	secondMomentArray = np.zeros(shape = (2*k+1, 2*k+1))
	secondMomentArray[(2 * k + 1) // 2, (2 * k + 1) // 2] = 1 / (k + 1) / (2 * k + 1)
	for i in range(secondMomentArray.shape[0]):
		for j in range(secondMomentArray.shape[1]):
			# Convert index in array to real values
			xval = i - (2 * k + 1) // 2
			yval = j - (2 * k + 1) // 2

			if xval == 0 or yval == 0:
				# Case at position (0, 0)
				if xval == yval: 
					continue
				# Case at position (i, 0) or (0, j)
				secondMomentArray[i, j] = 1 / 2 / (k + 1) / (2 * k + 1)
			
			elif xval == -yval or xval == yval:
				secondMomentArray[i, j] = 3 / 4 / (k + 1) / (2 * k + 1)
	
			else: 
				secondMomentArray[i, j] = 1 / 2 / (k + 1) / (2 * k + 1)

	return secondMomentArray

def getSecondMomentArrayDirichlet(alpha):
	k = len(alpha) // 2
	alpha_0 = np.sum(alpha)
	
	# The i, j element will be the covariance of X_i, X_j
	dirichletSecondMomentArray = np.zeros(shape = (len(alpha), len(alpha)))
	for i in range(dirichletSecondMomentArray.shape[0]):
		for j in range(dirichletSecondMomentArray.shape[1]):			
			alpha_i = alpha[i] / alpha_0 
			alpha_j = alpha[j] / alpha_0

			if i != j: 
				dirichletSecondMomentArray[i, j] += -alpha_i * alpha_j / (alpha_0 + 1) + alpha_i * alpha_j
			else: 
				dirichletSecondMomentArray[i, j] += alpha_i * (1 - alpha_i) / (alpha_0 + 1) + alpha_i**2
				
	return dirichletSecondMomentArray

def getSecondMomentArraySymmetricArbitraryAlpha(alpha):
	alpha = np.array(alpha)
	
	k = len(alpha) // 2
	dirichletArray = getSecondMomentArrayDirichlet(alpha)
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
	
			symmetricArray[i, j] = (dirichletArray[i, j] + dirichletArray[i_n, j] + dirichletArray[i, j_n] + dirichletArray[i_n, j_n]) / 4
			firstMomentArray[i, j] = firstMoments[i] * firstMoments[j]
	
	return firstMomentArray, symmetricArray

def getSecondMomentArrayBetaBinom():
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

def getSecondMomentSticky():
	pdf = np.array([0.01, 0, 0.99])
	pdf_r = np.array([0.99, 0, 0.01])

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

def getSecondMomentArrayThirdMoment():
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

def getSecondMomentArrayConstDiffCoeff(k):
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

def calculateCoefficentArbitraryDirichlet(k, alpha):
	""" Get extreme value coefficient for dirichlet distribution that is flipped: """
	firstMomentArray, secondMomentArray = getSecondMomentArraySymmetricArbitraryAlpha(alpha)
	localTimeCoeff = calculateLocalTimeSum(k, secondMomentArray, firstMomentArray)
	
	# Get variance of D
	varD = 0
	for idx_i in range(secondMomentArray.shape[0]):
		for idx_j in range(secondMomentArray.shape[1]):
			xval = idx_i - (2 * k + 1) // 2
			yval = idx_j - (2 * k + 1) // 2

			varD += xval**2 * yval**2 * secondMomentArray[idx_i, idx_j]
	
	meanArray = alpha / np.sum(alpha)
	meanArray = (meanArray + np.flip(meanArray)) / 2
	xvals = np.arange(-k, k+1)
	D = 1/2 * np.sum(meanArray * xvals**2)
	varD -= (2 * D)**2
	coeff = varD / localTimeCoeff
	return coeff, varD, D

def calculateCoefficientBetaBinom():
	firstMomentArray, secondMomentArray, meanArray = getSecondMomentArrayBetaBinom()
	k = firstMomentArray.shape[0] // 2
	localTimeCoeff = calculateLocalTimeSum(k, secondMomentArray, firstMomentArray)
	
	# Get variance of thirdMoment
	thirdMoment = 0
	for idx_i in range(secondMomentArray.shape[0]):
		for idx_j in range(secondMomentArray.shape[1]):
			xval = idx_i - (2 * k + 1) // 2
			yval = idx_j - (2 * k + 1) // 2

			thirdMoment += xval**3 * yval**3 * secondMomentArray[idx_i, idx_j]
	
	xvals = np.arange(-k, k+1)
	D = 1/2 * np.sum(meanArray * xvals**2)
	avgThirdMoment = np.sum(meanArray * xvals**3)

	thirdMoment -= (avgThirdMoment)**2
	coeff = thirdMoment / localTimeCoeff

	return coeff, thirdMoment, D

def calculateCoefficientConstDiffCoeff(k):
	firstMomentArray, secondMomentArray, meanArray = getSecondMomentArrayConstDiffCoeff(k)
	localTimeCoeff = calculateLocalTimeSum(k, secondMomentArray, firstMomentArray)
	
	# Get variance of fourth moment
	fourthMoment = 0
	for idx_i in range(secondMomentArray.shape[0]):
		for idx_j in range(secondMomentArray.shape[1]):
			xval = idx_i - (2 * k + 1) // 2
			yval = idx_j - (2 * k + 1) // 2

			fourthMoment += xval**4 * yval**4 * secondMomentArray[idx_i, idx_j]
	
	xvals = np.arange(-k, k+1)
	D = 1/2 * np.sum(meanArray * xvals**2)
	avgFourthMoment = np.sum(meanArray * xvals**4)

	fourthMoment -= (avgFourthMoment)**2
	coeff = fourthMoment / localTimeCoeff

	return coeff, fourthMoment, D

def calculateCoefficientThirdMoment():
	firstMomentArray, secondMomentArray, meanArray = getSecondMomentArrayThirdMoment()
	k = firstMomentArray.shape[0] // 2
	localTimeCoeff = calculateLocalTimeSum(k, secondMomentArray, firstMomentArray)
	
	# Get variance of thirdMoment
	thirdMoment = 0
	for idx_i in range(secondMomentArray.shape[0]):
		for idx_j in range(secondMomentArray.shape[1]):
			xval = idx_i - (2 * k + 1) // 2
			yval = idx_j - (2 * k + 1) // 2

			thirdMoment += xval**3 * yval**3 * secondMomentArray[idx_i, idx_j]
	
	xvals = np.arange(-k, k+1)
	D = 1/2 * np.sum(meanArray * xvals**2)
	avgThirdMoment = np.sum(meanArray * xvals**3)

	thirdMoment -= (avgThirdMoment)**2
	coeff = thirdMoment / localTimeCoeff

	return coeff, thirdMoment, D

if __name__ == '__main__':
	from matplotlib import pyplot as plt

	firstMomentArray, secondMomentArray, firstMoments = getSecondMomentSticky()
	mu = getInvMeasure(1001, secondMomentArray, firstMomentArray)
	
	fig, ax = plt.subplots()
	ax.set_yscale("log")
	ax.plot(mu)
	fig.savefig("InvMeasure.png")