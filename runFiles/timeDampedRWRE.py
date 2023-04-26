import numpy as np 
from numba import njit
import csv
import sys
import os

@njit
def iteratePDFTimeDamped(pdf, t, gamma=1/2):
	pdf_new = np.zeros(pdf.shape)
	damping = (t+1)**(gamma)
	biases = np.random.uniform(-1/2, 1/2, pdf.size) / damping

	pdf_new[0] = pdf[0] * (1/2 + biases[0])
	for i in range(1, t+2):
		pdf_new[i] = pdf[i] * (1/2 + biases[i]) + pdf[i-1] * (1/2 - biases[i-1])
	
	return pdf_new

@njit
def iteratePDFSpaceTimeDamped(pdf, t, gamma=1/2):
	pdf_new = np.zeros(pdf.shape)
	xvals = 2 * np.arange(0, pdf_new.size, step=1) - t
	damping = (t+1)**(gamma)
	biases = np.random.uniform(-1/2, 1/2, pdf.size) * (xvals / damping)

	# Handle edge cases
	pdf_new[0] = pdf[0] * (1/2 + biases[0])
	pdf_new[t+1] = pdf[t] * (1/2 - biases[t])
	
	for i in range(1, t+1):
		assert np.abs(biases[i]) <= 1/2
		pdf_new[i] = pdf[i] * (1/2 + biases[i]) + pdf[i-1] * (1/2 - biases[i-1])
	
	return pdf_new

@njit
def iteratePDFSpaceTimeDampedFull(pdf, t):
	pdf_new = np.zeros(pdf.shape)
	xvals = 2 * np.arange(0, pdf_new.size, step=1) - t
	biases = np.random.uniform(-1/2, 1/2, pdf.size) * (1 - (t - xvals) / (t + xvals)) * (t + xvals) / 2 / t
	
	# Divide by 0 when xvals = -t. Therefore, set equal to when xvals=t due to symmetry
	biases[0] = 0 

	# Handle edge cases
	pdf_new[0] = pdf[0] * (1/2 + biases[0])
	pdf_new[t+1] = pdf[t] * (1/2 - biases[t])
	
	for i in range(1, t+1):
		assert np.abs(biases[i]) <= 1/2
		pdf_new[i] = pdf[i] * (1/2 + biases[i]) + pdf[i-1] * (1/2 - biases[i-1])
	
	return pdf_new

def getVelocities(times, vs, save_files, gamma):
	# Set up csv writers for each file
	files = [open(i, 'a') for i in save_files]
	writers = [csv.writer(f) for f in files]

	# Write the headers to each file
	for i in range(len(writers)):
		writers[i].writerow(["Time", "Position", "logP"])
		files[i].flush()

	# Initialize PDF and time
	pdf = np.zeros(max(times)+1)

	# Divide by 0 error when t=0 so go to first time step
	pdf[0] = 0.5
	pdf[1] = 0.5
	t = 1

	for _ in range(max(times)): 
		pdf = iteratePDFSpaceTimeDampedFull(pdf, t)
		assert np.abs(np.sum(pdf) - 1) < 1e-10, np.sum(pdf)
		t += 1

		if t in times: 
			xvals = np.floor(vs * t **(3/4)) 
			idx = np.ceil((xvals + t) / 2).astype(int) # need to account for shifting index
			
			for i in range(len(xvals)): 
				prob = np.sum(pdf[idx[i]:])
				writers[i].writerow([t, xvals[i], np.log(prob).astype(float)])
				files[i].flush()

	for f in files:
		f.close()

if __name__ == '__main__':
	#topDir, sysID, tMax, gamma = '.', '0', '1e5', '1'
	(topDir, sysID, tMax, gamma) = sys.argv[1:]
	tMax = int(float(tMax))
	gamma = float(gamma)
	vs = np.arange(0.2, 1, 0.1)
	times = np.geomspace(1, tMax, 5000).astype(int)
	times = np.unique(times)
	
	save_dirs = [os.path.join(topDir, str(v)[:3]) for v in vs]
	# Make save directories once if system id is 0
	if int(sysID) == 0:
		for save_dir in save_dirs:
			os.makedirs(save_dir, exist_ok=True)

	# only keep a single digit of precision in save directory
	save_files = [os.path.join(save_dir, f'Velocities{sysID}.txt') for save_dir in save_dirs]
	
	getVelocities(times, vs, save_files, gamma)