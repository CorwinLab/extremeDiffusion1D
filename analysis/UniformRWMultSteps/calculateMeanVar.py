import numpy as np 
import glob 
import os 
import pandas as pd

def getMeanVar(files, tMax):
	first_moment = None
	second_moment = None
	num = 0
	
	for i, f in enumerate(files[1:]):
		try:
			data = np.loadtxt(f, skiprows=1, delimiter=',')
		except Exception as e:
			print(e, f)
			continue

		# Check if time is less than maximum time
		if data[-1, 0] < tMax:
			continue
		data = data[data[:, 0] < tMax]
		time = data[:, 0]
		
		# Need to take the log of the probability distribution
		data[:, 2:] = np.log(data[:, 2:])

		if first_moment is None:
			first_moment = data[:, 1:]
		else:
			first_moment += data[:, 1:]
		if second_moment is None:
			second_moment = data[:, 1:] ** 2
		else:
			second_moment += data[:, 1:] ** 2
		num += 1
		print(f)

	mean = first_moment / num
	var = second_moment / num - mean**2
	time = time.reshape(len(time), 1)
	mean = np.hstack((time, mean))
	var = np.hstack((time, var))
	print('Number of Files:', num)
	return mean, var

dir = '/home/jacob/Desktop/talapasMount/JacobData/UniformRWMultSteps/'
folders = os.listdir(dir)
tMax = 100_000

for f in folders: 
	files = glob.glob(os.path.join(dir, f, 'Q*.txt'))

	mean, var = getMeanVar(files, tMax)
	mean_file = os.path.join(dir, f, 'Mean.txt')
	var_file = os.path.join(dir, f, 'Var.txt')
	df = pd.read_csv(files[0])

	mean = pd.DataFrame(mean, columns=df.columns)
	var = pd.DataFrame(var, columns=df.columns)

	mean.to_csv(mean_file, index=False)
	var.to_csv(var_file, index=False)