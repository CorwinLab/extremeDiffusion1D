import numpy as np 
from matplotlib import pyplot as plt
import glob 
import os 

def getFinalProbs(files, tMax):
	probs = []
	for f in files:
		data = np.loadtxt(f, skiprows=1, delimiter=',')
		if data[-1, 0] < tMax:
			continue 
		data = data[data[:, 0] < tMax]
		probs.append(data[-1, 2])
		t = data[-1, 0]
		print(f)
	return t, probs

dir = '/home/jacob/Desktop/talapasMount/JacobData/ScatteringVelocities'
vs = os.listdir(dir)

for v in vs:
	files = glob.glob(os.path.join(dir, v, "Q*.txt"))
	t, probs = getFinalProbs(files, 2*10**4)
	
	probs_file = os.path.join(dir, v, "Probs.txt")
	t_file = os.path.join(dir, v, "T.txt")
	
	np.savetxt(probs_file, probs)
	np.savetxt(t_file, [t])