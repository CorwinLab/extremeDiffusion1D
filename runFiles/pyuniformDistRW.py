import numpy as np
import csv
from numba import njit 

@njit
def iteratePDF(pdf, max_step_size=3):
	pdf_new = np.zeros(pdf.size)

	for idx in range(max_step_size, len(pdf_new)-max_step_size):
		# Step up transition kernel
		width = np.random.randint(0, max_step_size + 1)
		
		transition_prob = np.ones(shape=2*width + 1) / (2*width + 1)
		
		pdf_new[idx - width: idx + width + 1] += transition_prob * pdf[idx]

	return pdf_new

@njit 
def getQuantile(N, pdf, max_idx):
	cdf = 0
	for idx in np.flip(np.arange(0, max_idx)):
		cdf += pdf[idx]
		if cdf >= 1/N:
			# Need to account for center of pdf
			return idx - pdf.size // 2
		
def getProbAtPos(pdf, xs):
	indeces = xs + pdf.size // 2
	probs = [pdf[idx] for idx in indeces]
	return probs

def evolveAndMeasureQuantileVelocity(tMax, max_step_size, N, v, save_file):
	# Get save times 
	times = np.unique(np.geomspace(1, tMax, 2500).astype(int))
	maxTime = np.max(times)

	# Initialize the probability distribution
	size = maxTime * (max_step_size + 1)
	
	# Need to enforce that the size is and odd number 
	# So that 0 is included
	if (size % 2) != 0:
		size += 1
	pdf = np.zeros(size)
	pdf[pdf.size // 2] = 1
	t = 0

	# Initialize save file writer
	f = open(save_file, "a")
	writer = csv.writer(f)
	writer.writerow(["Time", "Quantile"] + v)
	f.flush()

	while t <= maxTime:
		# First find indeces of array which are first/last nonzer 
		nonzeros = np.nonzero(pdf)[0]
		min_idx, max_idx = nonzeros[0], nonzeros[-1]

		# Slice part of array that is nonzero with padding of max_step_size 
		# on both sides
		pdf_pass = pdf[min_idx - max_step_size: max_idx + max_step_size + 1]
		pdf_pass = iteratePDF(pdf_pass, max_step_size=max_step_size)

		# Insert iterated pdf into original pdf
		pdf[min_idx - max_step_size: max_idx + max_step_size + 1] = pdf_pass 
		t+=1

		if t in times: 
			quantile = getQuantile(N, pdf, max_idx)
			x = (np.array(v) * t**(3/4)).astype(int)
			prob = getProbAtPos(pdf, x)
			
			# Need to get probabilities for all x values 
			row = [t, quantile] + prob 
			writer.writerow(row)
			
			f.flush()

if __name__ == '__main__':
	from matplotlib import pyplot as plt 

	tMax = 500
	xvals = np.arange(-tMax, tMax+1)
	pdf = np.zeros(xvals.size)
	pdf[pdf.size//2] = 1
	max_step_size = 5

	tEvolve = tMax // (max_step_size+1)
	min_idx = pdf.size // 2
	max_idx = pdf.size // 2 + 1

	for t in range(tEvolve):
		nonzeros = np.nonzero(pdf)[0]
		min_idx, max_idx = nonzeros[0], nonzeros[-1]
		pdf_pass = pdf[min_idx - max_step_size: max_idx + max_step_size + 1]
		pdf_pass = iteratePDF(pdf_pass, max_step_size=max_step_size)
		pdf[min_idx - max_step_size: max_idx + max_step_size + 1] = pdf_pass 

	n = tEvolve * (max_step_size+1)
	p = 0.5
	Nexp = 12
	N = float(f"1e{Nexp}")
	quantile = getQuantile(N, pdf, max_idx)

	x = 50
	prob = getProbAtPos(pdf, x)

	fig, ax = plt.subplots()
	ax.set_ylim([10**-20, 10**0])
	ax.set_xlim([-200, 200])
	ax.set_yscale("log")
	ax.plot(xvals, pdf)
	ax.vlines(quantile, 10**-50, 1, color='r', ls='--', label=f'N=1e{Nexp}')
	ax.scatter(x, prob, c='m', zorder=3, label='Prob at x=50')
	ax.set_xlabel("x")
	ax.set_ylabel("p(x,t)")
	ax.legend()
	fig.savefig("Dist.png")
