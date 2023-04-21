import numpy as np 
import npquad 
from pyDiffusion import ScatteringModel
import sys
import os

if __name__ == '__main__':
	#topDir, sysID, tMax = './', '0', '1e5'
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

	s = ScatteringModel('beta', [1, 1], tMax)
	s.getVelocities(times, vs, save_files)