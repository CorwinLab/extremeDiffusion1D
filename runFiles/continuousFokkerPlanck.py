import numpy as np
from matplotlib import pyplot as plt
from numba import njit 
import csv
import mpmath

@njit 
def forwardEuler(f, dx, dt, D0, sigma):
	'''
	Example:
	
	D0 = 1
	sigma = 0.1
	dx = 0.1
	dt = dx**2 / 2 / D0 / 4
	L = 10000
	xvals = np.round(np.arange(-L, L+dx, step=dx), 1)
	
	p = np.zeros(xvals.shape)
	p = 1 / np.sqrt(2 * np.pi * (4 * D0)) * np.exp(-1/2 * (xvals)**2 / (4 * D0))
	p /= np.sum(p)
	N = 1e50
	
	tMax = 5000
	quantile = np.zeros(len(range(tMax)))
	for t in range(tMax):
		p = forwardEuler(p, dx, dt, D0, sigma, 1, len(p)-1)
		xmeasure = int(1 / 10 * t)
		prob = getProbAtX(p, xvals, xmeasure)
		idx = getQuantile(p, N)
		quantile[t] = xvals[idx]

	fig, ax = plt.subplots()
	ax.plot(xvals, p)
	ax.set_yscale("log")
	ax.set_xlim([-1000, 1000])
	ax.scatter(xmeasure, prob, c='k')
	fig.savefig("ProbDist.png")

	fig, ax = plt.subplots()
	ax.plot(range(tMax), quantile)
	ax.set_xscale("log")
	ax.set_yscale("log")
	fig.savefig("Quantile.png")
	'''
	
	f_new = np.zeros(f.shape)
	Ds = np.random.normal(loc = D0, scale=sigma, size=f_new.size)
	# Ds = np.random.lognormal(mean = D0, sigma = sigma, size=f_new.size)
	assert np.all(Ds >= 0)

	field_times_d = Ds * f
	for i in range(1, len(f_new)-1):
		f_new[i] = dt / dx**2 * (field_times_d[i+1] - 2 * field_times_d[i] + field_times_d[i-1]) + f[i]

	return f_new

def getProbAtX(prob, xvals, x):
	idx = np.where(xvals == x)[0][0]
	return prob[idx]

@njit
def getQuantile(prob, N):
	cdf = 0
	for idx in range(prob.size, 0, -1):
		cdf += prob[idx]
		if cdf >= 1/N:
			return idx
		
def evolveAndGetProbs(tMax, v, D0, sigma, dx, save_file):
	'''
	Examples

	'''

	# Specify what times to go to
	times = np.unique(np.geomspace(1, tMax, 500).astype(int))
	maxTime = np.max(times)

	# Set time-scale
	dt = dx**2 / 2 / D0 / 4
	t = dt
	L = maxTime * 2

	# Set x-scale to precision within one decimal place
	xvals = np.round(np.arange(-L, L+dx, step=dx), 1)
	
	# Initialize probability distribution
	# Initialized to Gaussian at time t=dt
	p = np.zeros(xvals.shape)
	p = 1 / np.sqrt(2 * np.pi * (4 * D0 * dt)) * np.exp(-1/2 * (xvals)**2 / (4 * D0 * dt))
	p /= np.sum(p)
	# grab min and max indeces
	nonzeros = np.nonzero(p)[0]
	min_idx, max_idx = nonzeros[0], nonzeros[-1]

	# Initialize save_file writer
	f = open(save_file, "a")
	writer = csv.writer(f)
	writer.writerow(["Time", "Position", "Probability"])
	f.flush()
	
	while t < maxTime:
		# Get indeces of array that are nonzero
		nonzeros = np.nonzero(p)[0]
		min_idx, max_idx = nonzeros[0], nonzeros[-1]
		
		# Get section of array that is nonzero and padded with 
		# two zeros on each side
		p_pass = p[min_idx-2:max_idx+3]
		p_new = forwardEuler(p_pass, dx, dt, D0, sigma)
		
		# Assign slice of array back in
		p[min_idx-2:max_idx+3] = p_new

		t += dt 
		if round(t, 10) in times:
			xmeasured  = int(v * t)
			probability = getProbAtX(p, xvals, xmeasured)
			writer.writerow([t, xmeasured, probability])
			f.flush()

def getMeanVarN(p, xvals, N):
	cdfN = (np.cumsum(p))**N
	pdfN = np.diff(cdfN)

	mean = np.sum(xvals[1:] * pdfN) 
	var = np.sum(xvals[1:] ** 2 * pdfN)- mean**2
	return mean, var

def evolveAndGetQuantiles(tMax, N, D0, sigma, dx, save_file):

	# Specify what times to go to
	times = np.unique(np.geomspace(1, tMax, 500).astype(int))
	maxTime = np.max(times)

	# Set time-scalep_pass = p[min_idx-2:max_idx+3]
	dt = dx**2 / 2 / D0 / 2
	t = dt
	L = maxTime * 2

	# Set x-scale to precision within one decimal place
	xvals = np.round(np.arange(-L, L+dx, step=dx), 1)
	
	# Initialize probability distribution
	# Initialized to Gaussian at time t=dt
	p = np.zeros(xvals.shape)
	p = 1 / np.sqrt(2 * np.pi * (4 * D0 * dt)) * np.exp(-1/2 * (xvals)**2 / (4 * D0 * dt))
	p /= np.sum(p)
	# grab min and max indeces
	nonzeros = np.nonzero(p)[0]
	min_idx, max_idx = nonzeros[0], nonzeros[-1]

	# Initialize save_file writer
	f = open(save_file, "a")
	writer = csv.writer(f)
	writer.writerow(["Time", "Mean", "Variance"])
	f.flush()
	
	while t < maxTime:
		# Get indeces of array that are nonzero
		nonzeros = np.nonzero(p)[0]
		min_idx, max_idx = nonzeros[0], nonzeros[-1]
		
		# Get section of array that is nonzero and padded with 
		# two zeros on each side
		p_pass = p[min_idx-2:max_idx+3]
		p_new = forwardEuler(p_pass, dx, dt, D0, sigma)
		
		# Assign slice of array back in
		p[min_idx-2:max_idx+3] = p_new

		t += dt 
		if round(t, 10) in times:
			mean, var = getMeanVarN(p, xvals, N)
			writer.writerow([t, mean, var])
			f.flush()

if __name__ == '__main__':
	tMax = 50000
	N = 1e10
	D0 = 0.1
	sigma = 0.01
	dx = 0.05
	save_file = './Probability0.txt'
	evolveAndGetQuantiles(tMax, N, D0, sigma, dx, save_file)