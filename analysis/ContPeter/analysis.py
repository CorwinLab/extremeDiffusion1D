import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 
import glob 
import os

rwre_model = "/home/jacob/Desktop/talapasMount/JacobData/ContPeter/1/"
ssrw_model = "/home/jacob/Desktop/talapasMount/JacobData/ContPeter/1Classical/"

rwre_files = glob.glob(os.path.join(rwre_model, "MaxPositions*.txt"))

maxTime = 50
avg_pos_sq = None
avg_pos = None
num = 0
for f in rwre_files: 
	df = pd.read_csv(f)
	df = df[df['Time'] <= maxTime]
	if max(df['Time']) < maxTime:
		continue
	if avg_pos is None: 
		avg_pos = df['Position'].values
	else: 
		avg_pos += df['Position'].values 

	if avg_pos_sq is None: 
		avg_pos_sq = df['Position'].values ** 2
	else:
		avg_pos_sq += df['Position'].values ** 2
	num += 1
	time = df['Time'].values

print(num)
avg = avg_pos / num 
var = avg_pos_sq / num - avg**2
rwre_df = pd.DataFrame(np.array([time, avg, var]).T, columns=["Time", "Mean", "Variance"])

ssrw_files = glob.glob(os.path.join(ssrw_model, "MaxPositions*.txt"))

maxTime = 50
avg_pos_sq = None
avg_pos = None
num = 0
for f in ssrw_files: 
	df = pd.read_csv(f)
	df = df[df['Time'] <= maxTime]
	if max(df['Time']) < maxTime:
		continue
	if avg_pos is None: 
		avg_pos = df['Position'].values
	else: 
		avg_pos += df['Position'].values 

	if avg_pos_sq is None: 
		avg_pos_sq = df['Position'].values ** 2
	else:
		avg_pos_sq += df['Position'].values ** 2
	num += 1
	time = df['Time'].values

avg = avg_pos / num 
var = avg_pos_sq / num - avg**2
ssrw_df = pd.DataFrame(np.array([time, avg, var]).T, columns=["Time", "Mean", "Variance"])

D = 1
xi = 1
N = 100000

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.plot(rwre_df['Time'].values, rwre_df['Mean'], label='RWRE')
ax.plot(ssrw_df['Time'].values, ssrw_df['Mean'], label='SSRW')
ax.legend()
fig.savefig("Mean.pdf", bbox_inches='tight')

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.plot(rwre_df['Time'].values, rwre_df['Variance'], label='RWRE')
ax.plot(ssrw_df['Time'].values, ssrw_df['Variance'], label='SSRW')
ax.legend()
fig.savefig("Var.pdf", bbox_inches='tight')