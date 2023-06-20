import numpy as np 
from matplotlib import pyplot as plt
import glob 
import os 
import pandas as pd

def getMeanVar(files, max_time):
	first_moment = None
	second_moment = None
	num = 0
	
	for i, f in enumerate(files[1:]): 
		data = pd.read_csv(f)
		if max(data['Time'].values) <= max_time:
			continue

		data = data[data['Time'] <= max_time]
		time = data['Time'].values
		data = data.values

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
	return mean, var, time

dir = '/home/jacob/Desktop/talapasMount/JacobData/ScatteringSweepLongest/0.01'
beta = 0.01
max_time = 500_000

files = glob.glob(os.path.join(dir, "Q*.txt"))
mean, var, time = getMeanVar(files, max_time)

mean_file = os.path.join(dir, "Mean.txt")
var_file = os.path.join(dir, "Var.txt")
time_file = os.path.join(dir, "Time.txt")

np.savetxt(mean_file, mean)
np.savetxt(var_file, var)
np.savetxt(time_file, time)