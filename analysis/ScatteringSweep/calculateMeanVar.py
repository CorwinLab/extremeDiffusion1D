import numpy as np 
from matplotlib import pyplot as plt
import glob 
import os 

def getMeanVar(files):
	first_data_set = np.loadtxt(files[0], skiprows=1, delimiter=',')
	first_moment = first_data_set[:, 1]
	second_moment = first_data_set[:, 1] ** 2
	num = 1

	for i, f in enumerate(files[1:]): 
		data = np.loadtxt(f, skiprows=1, delimiter=',')
		first_moment += data[:, 1]
		second_moment += data[:, 1] ** 2
		num += 1
		print(i)

	mean = first_moment / num
	var = second_moment / num - mean**2
	return mean, var, data[:, 0]

dir = '/home/jacob/Desktop/talapasMount/JacobData/ScatteringSweep'
betas = os.listdir(dir)

for b in betas: 
    files = glob.glob(os.path.join(dir, b, "Q*.txt"))
    mean, var, time = getMeanVar(files)
    
    mean_file = os.path.join(dir, b, "Mean.txt")
    var_file = os.path.join(dir, b, "Var.txt")
    time_file = os.path.join(dir, b, "Time.txt")
    
    np.savetxt(mean_file, mean)
    np.savetxt(var_file, var)
    np.savetxt(time_file, time)