import numpy as np
from scipy.stats import binom
from numba import njit, vectorize
from math import gamma
import csv

@vectorize
def binomialCoefficient(n, k):
	return gamma(n+1) / (gamma(k+1) * gamma(n-k+1))

@njit 
def binomialDist(xvals, n, p):
	return p**xvals * (1-p)**(n-xvals) * binomialCoefficient(n, xvals)

@njit
def iteratePDF(pdf, min_idx, max_idx, max_step_size=3):
	pdf_new = np.zeros(pdf.size)

	for idx in range(min_idx, max_idx+1):
		# Step up transition kernel
		# step_size = np.random.binomial(max_step_size, 0.5) + 1
		step_size = np.random.choice(np.array([max_step_size/2 - 1, max_step_size/2, max_step_size/2+1]))
		
		xvals = np.arange(0, 2 * step_size + 1)
		
		transition_prob = binomialDist(xvals, 2*step_size, 0.5)

		# The version below can't be numba compiled 
		# transition_prob = binom.pmf(xvals, 2 * step_size, 0.5)

		# The transition probability kernel is always *just* bigger than 1
		# because of precision issues. Dividing by it's sum helps keep the 
		# kernel normalized. Non-normalized kernels means the pdf doesn't 
		# sum to 1
		transition_prob = transition_prob / np.sum(transition_prob)

		current_kernel_size = transition_prob.size
		
		pdf_new[idx - current_kernel_size // 2: idx + current_kernel_size // 2 + 1] += transition_prob * pdf[idx]

	nonzeros = np.nonzero(pdf_new)[0]
	min_idx, max_idx = nonzeros[0], nonzeros[-1]

	return pdf_new, min_idx, max_idx

@njit 
def getQuantile(N, pdf, max_idx):
	cdf = 0
	for idx in np.flip(np.arange(0, max_idx)):
		cdf += pdf[idx]
		if cdf >= 1/N:
			# Need to account for center of pdf
			return idx - pdf.size // 2
		
def getProbAtPos(pdf, x):
	idx = x + pdf.size // 2
	return pdf[idx]

def evolveAndMeasureQuantileVelocity(tMax, max_step_size, N, v, save_file, save_pdf):
	# Get save times 
	times = np.unique(np.geomspace(1, tMax, 2500).astype(int))
	maxTime = np.max(times)

	# Initialize the probability distribution
	size = maxTime * (max_step_size + 1)
	# Need to enforce that the size is and odd number
	if (size % 2) != 0:
		size += 1
	pdf = np.zeros(size)
	pdf[pdf.size // 2] = 1
	t = 0

	min_idx = pdf.size // 2
	max_idx = pdf.size // 2 + 1

	# Initialize save file writer
	f = open(save_file, "a")
	writer = csv.writer(f)
	writer.writerow(["Time", "Quantile", "Probability"])
	f.flush()

	while t <= maxTime:
		pdf, min_idx, max_idx = iteratePDF(pdf, min_idx, max_idx, max_step_size)
		t+=1

		if t in times: 
			quantile = getQuantile(N, pdf, max_idx)
			x = int(v * t**(3/4))
			prob = getProbAtPos(pdf, x)
			writer.writerow([t, quantile, prob])
			f.flush()
	np.savetxt(save_pdf, pdf)

if __name__ == '__main__':
	from matplotlib import pyplot as plt 

	tMax = 5000
	xvals = np.arange(-tMax, tMax+1)
	pdf = np.zeros(xvals.size)
	pdf[pdf.size//2] = 1
	max_step_size = 100

	tEvolve = tMax // (max_step_size+1)
	min_idx = pdf.size // 2
	max_idx = pdf.size // 2 + 1
	
	step_size = np.random.binomial(max_step_size, 0.5, size=10000000) + 1
	fig, ax = plt.subplots()
	ax.hist(step_size,bins=100)
	fig.savefig("StepSize.png")

	for t in range(tEvolve):
		pdf, min_idx, max_idx = iteratePDF(pdf, min_idx, max_idx, max_step_size=max_step_size)
		print(t)

	n = tEvolve * (max_step_size+1)
	p = 0.5
	theory = binom.pmf(np.arange(0, n+1), n, p)
	Nexp = 12
	N = float(f"1e{Nexp}")
	quantile = getQuantile(N, pdf, max_idx)

	x = 200
	prob = getProbAtPos(pdf, x)

	fig, ax = plt.subplots()
	ax.set_yscale("log")
	ax.plot(xvals, pdf)
	ax.set_ylim([10**-20, 10**0])
	ax.set_xlim([-500, 500])
	ax.plot(np.arange(-n // 2, n // 2 + 1), theory, c='k', ls='--', label='Mean Binomial')
	ax.vlines(quantile, 10**-50, 1, color='r', ls='--', label=f'N={Nexp}')
	ax.scatter(x, prob, c='m', zorder=3, label='Prob at x=200')
	ax.set_xlabel("x")
	ax.set_ylabel("p(x,t)")
	ax.legend()
	fig.savefig("Dist.png")
