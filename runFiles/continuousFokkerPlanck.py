import numpy as np
from matplotlib import pyplot as plt
from numba import njit 
import csv

@njit 
def forwardEuler(f, dx, dt, D0, sigma, minIdx, maxIdx):
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
	Ds = np.zeros(f_new.shape)
	Ds[minIdx-1:maxIdx +1] = np.random.normal(loc = D0, scale=sigma, size=maxIdx - minIdx + 2)
	assert np.all(Ds[minIdx:maxIdx+1] >= 0)

	field_times_d = Ds * f
	for i in range(minIdx-1, maxIdx+2):
		f_new[i] = dt / dx**2 * (field_times_d[i+1] - 2 * field_times_d[i] + field_times_d[i-1]) + f[i]

	nonzeros = np.nonzero(f_new)[0]
	min_idx, max_idx = nonzeros[0], nonzeros[-1]

	return f_new, min_idx, max_idx

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
	tMax = 5000
	v = 1/5
	D0 = 1
	sigma = 0.1
	dx = 1
	save_file = './Probability.txt'
	evolveAndGetProbs(tMax, v, D0, sigma, dx, save_file)
	'''

	# Specify what times to go to
	times = np.unique(np.geomspace(1, tMax, 500).astype(int))
	maxTime = np.max(times)

	# Set time-scale
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
	writer.writerow(["Time", "Position", "Probability"])
	f.flush()
	
	while t < maxTime:
		# This could be incredibly faster if we only pass the nonzero part of the array
		# I'm just going to continu for now though.
		p, min_idx, max_idx = forwardEuler(p, dx, dt, D0, sigma, min_idx, max_idx)
		assert (min_idx > 0) and (max_idx < (len(p)-1))
		t += dt 
		if t in times:
			xmeasured  = int(v * t)
			probability = getProbAtX(p, xvals, xmeasured)
			writer.writerow([t, xmeasured, probability])
			f.flush()
