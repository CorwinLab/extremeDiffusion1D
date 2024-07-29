import numpy as np
from scipy.sparse import linalg

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
	# Could probably combine the first and second sums but that's okay.
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

 		# sumOverij = sumOverij / (2 * k + 1)**2

		# Calculate the invariance measure ratio - only works if obeys detail balance
		# invMeasurel = 0 
		# invMeasure0 = 0
		# mean_vals = 1 / (2 * k + 1)
		# for i in range(secondMomentArray.shape[0]):
		# 	for j in range(secondMomentArray.shape[1]):
		# 		xval = i - (2 * k + 1) // 2
		# 		yval = j - (2 * k + 1) // 2
				
		# 		if np.abs(xval - yval) == l:
		# 			invMeasurel += secondMomentArray[i, j]
		# 			invMeasure0 += mean_vals ** 2

		# measure = 2 * invMeasurel / invMeasure0
		# secondSum += sumOverij * measure
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
	return coeff, varD

# def getInvMeasure(l, secondMomentArray, mean_vals, k):
# 	invMeasurel = 0 
# 	invMeasure0 = 0
# 	for i in range(secondMomentArray.shape[0]):
# 		for j in range(secondMomentArray.shape[1]):
# 			xval = i - (2 * k + 1) // 2
# 			yval = j - (2 * k + 1) // 2

# 			if np.abs(xval - yval) == l:
# 				invMeasurel += secondMomentArray[i, j]
# 				invMeasure0 += mean_vals[i] * mean_vals[j]

# 	return 2 * invMeasurel / invMeasure0

if __name__ == '__main__':
	alpha = 0.01
	size = 3
	alpha = np.ones(size) * alpha
	k = size // 2
	D = 1 / 2 / 3 * k * (k + 1)
	firstMomentArray, symmetricArray = getSecondMomentArraySymmetricArbitraryAlpha(alpha)
	localTimeCoeff = calculateLocalTimeSum(k, symmetricArray, firstMomentArray)
	print(localTimeCoeff, 4 * D, (4 * D - localTimeCoeff) / localTimeCoeff * 100)

	# alpha = np.array([5, 10, 5])
	# k = len(alpha) // 2
	# sigma2 = 0 
	# for i in range(-k, k+1):
	# 	sigma2 += alpha[i] / np.sum(alpha) * i**2
	# D = sigma2 / 2
	# arr = getSecondMomentArraySymmetricArbitraryAlpha(alpha)
	# localTimeCoeff = calculateLocalTimeSum(k, arr)
	# coeff, varD = calculateCoefficentArbitraryDirichlet(k, alpha)
	# print(localTimeCoeff, 4 * D)