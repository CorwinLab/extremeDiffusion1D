import numpy as np 
from matplotlib import pyplot as plt
import glob 
import os 

def getMeanVar(files):
	first_moment = None
	second_moment = None
	num = 0
	
	for i, f in enumerate(files[1:]): 
		data = np.loadtxt(f, skiprows=1, delimiter=',')
		if data[-1, 0] < 99885:
			continue
		if first_moment is None:
			first_moment = data[:, 1]
		else:
			first_moment += data[:, 1]
		if second_moment is None:
			second_moment = data[:, 1] ** 2
		else:
			second_moment += data[:, 1] ** 2
		num += 1
		print(i)

	mean = first_moment / num
	var = second_moment / num - mean**2
	return mean, var, data[:, 0]

# Calculate Mean and Variance for this directory 
dir = '/home/jacob/Desktop/talapasMount/JacobData/GeneralizedScattering/'
files = glob.glob(os.path.join(dir, "Q*.txt"))
mean, var, time = getMeanVar(files)

mean_file = os.path.join(dir, "Mean.txt")
var_file = os.path.join(dir, "Var.txt")
time_file = os.path.join(dir, "Time.txt")

np.savetxt(mean_file, mean)
np.savetxt(var_file, var)
np.savetxt(time_file, time)