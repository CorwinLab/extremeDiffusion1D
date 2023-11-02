import numpy as np
import npquad
import os 
import sys 
from pyDiffusion import DiffusionTimeCDF
import csv 

def getVelocities(d, times, vs, save_files):
	# Set up csv writers for each file
	files = [open(i, 'a') for i in save_files]
	writers = [csv.writer(f) for f in files]

	# Write the headers to each file
	for i in range(len(writers)):
		writers[i].writerow(["Time", "Position", "logP"])
		files[i].flush()

	for t in times: 
		d.evolveToTime(t)
		xvals = np.floor(vs * d.getTime() **(3/4)) 
		idx = np.ceil((xvals + d.getTime()) / 2).astype(int) # need to account for shifting index
		
		for i in range(len(xvals)): 
			prob = d.getProbAtX(idx[i])
			writers[i].writerow([d.getTime(), xvals[i], np.log(prob).astype(float)])
			files[i].flush()

	for f in files:
		f.close()

if __name__ == '__main__':
	#topDir, sysID, tMax = '.', '0', '1e4'
	(topDir, sysID, tMax) = sys.argv[1:]
	tMax = int(float(tMax))
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
	
	d = DiffusionTimeCDF('beta', [1, 1], times[-1])
	getVelocities(d, times, vs, save_files)
