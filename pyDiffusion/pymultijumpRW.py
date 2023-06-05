import numpy as np
from matplotlib import pyplot as plt
from numba import njit
import csv 

@njit 
def randomDirichlet(size):
	rand_vals = np.random.uniform(0, 1, size)
	return rand_vals / np.sum(rand_vals)

@njit
def iterateTimeStep(pdf, t, step_size=3):
	pdf_new = np.zeros(pdf.size)
	
	# I'm not entirely sure how/why but using this end point means
	# that we iterate over the entire array but no further
	for i in range(0, t * (step_size-1) - step_size + 2):
		rand_vals = randomDirichlet(step_size)
		pdf_new[i: i + step_size] += rand_vals * pdf[i]

	return pdf_new

def measureProbAtPosition(pdf, x, t, step_size):
	center = t * (step_size // 2)
	idx = x + center 
	return pdf[idx]

@njit	
def measureQuantile(pdf, N, t, step_size):
	cdf = 0 
	for i in range(pdf.size):
		cdf += pdf[i]
		if cdf >= 1/N:
			center = t * (step_size // 2)
			return i - center  

def evolveAndMeasureQuantileVelocity(tMax, step_size, N, v, save_file):
	# Ensure the step_size is odd 
	assert (step_size % 2) != 0, f"Step size is not and odd number but {step_size}"

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
	writer.writerow(["Time", "Quantile", "Probability"])
	f.flush()
	
	maxTime = np.max(times)
	while t < maxTime: 
		pdf = iterateTimeStep(pdf, t+1, step_size)
		
		t+=1

		if t in times: 
			quantile = measureQuantile(pdf, N, t, step_size)
			x = int(v * t**(3/4))
			prob = measureProbAtPosition(pdf, x, t, step_size)
			writer.writerow([t, np.abs(quantile), prob])

if __name__ == '__main__':
	L = 10000
	step_size = 11
	tMax = L // step_size

	# This sets the size to a little bigger than the required
	# size to iterate to tMax.
	size = tMax * step_size
	pdf = np.zeros(size)
	pdf[0] = 1
	t = 0 

	while t < tMax:
		#print(f"t={t}, {np.sum(pdf)}")
		pdf = iterateTimeStep(pdf, t+1, step_size)
		t += 1

	# This gives the correct xvalues for the array
	# note that it only works when the step size is 
	# odd. If it's even things get messed up.
	center = t * (step_size // 2)
	xvals = np.arange(0, pdf.size) - center
	
	# Make sure the measurements are correct
	xMeasurement = 500
	pMeasurement = measureProbAtPosition(pdf, xMeasurement, t, step_size)

	N = 1e12
	quantile = measureQuantile(pdf, N, t, step_size)

	fig, ax = plt.subplots()
	ax.set_yscale("log")
	ax.set_ylim([10**-20, 10**-2])
	ax.set_xlim([-1500, 1500])
	ax.plot(xvals, pdf)
	ax.scatter(xMeasurement, pMeasurement, c='k', zorder=3)
	ax.vlines(quantile, 10**-20, 1, color='r', ls='--')
	ax.set_xlabel("x")
	ax.set_ylabel(r"$p_{\bf{B}}(x,t)$")
	fig.savefig("DirichletDist.png")

	# Test evolve and get quantile and velocity function
	tMax = 1000
	times = np.unique(np.geomspace(1, tMax, num = 100).astype(int))
	step_size = 11
	N = 1e12 
	v = 1/2 
	save_file = 'Quantiles.txt'

	evolveAndMeasureQuantileVelocity(times, step_size, N, v, save_file)