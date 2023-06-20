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
		# Check if time is less than maximum time
		if data[-1, 0] < 99885:
			continue
		
		time = data[:, 0]

		if first_moment is None:
			first_moment = np.log(1-data[:, 2])
		else:
			first_moment += np.log(1-data[:, 2]) 
		if second_moment is None:
			second_moment = np.log(1-data[:, 2]) ** 2
		else:
			second_moment += np.log(1-data[:, 2]) ** 2
		num += 1
		print(f)

	mean = first_moment / num
	var = second_moment / num - mean**2
	mean = np.vstack((time, mean)).T
	var = np.vstack((time, var)).T
	return mean, var

# Calculate Mean and Variance for this directory 
dir = '/home/jacob/Desktop/talapasMount/JacobData/ScatteringVelocities'
vs = os.listdir(dir)

for v in vs:
    files = glob.glob(os.path.join(dir, v, "Q*.txt"))
    mean, var = getMeanVar(files)
    
    mean_file = os.path.join(dir, v, "Mean.txt")
    var_file = os.path.join(dir, v, "Var.txt")
    
    np.savetxt(mean_file, mean)
    np.savetxt(var_file, var)