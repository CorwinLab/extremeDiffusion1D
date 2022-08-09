import os 
import glob 
import sys 
sys.path.append("../../pysrc")
import numpy as np
import npquad
import pandas as pd
from matplotlib import pyplot as plt

dir = "/home/jacob/Desktop/talapasMount/JacobData/LongFirstPassageCDF/F*.txt"
files = glob.glob(dir)
average_data = None
average_data_squared = None
number_of_files = 0
max_dist = 16000
for f in files:
    data = pd.read_csv(f, delimiter=',') # columns are distance, mean, variance, quantile position
    if max(data['distance']) < max_dist: 
        continue
    data = data[data['distance'] <= max_dist].values
    number_of_files += 1
    if average_data is None: 
        average_data = data
        average_data_squared = data ** 2
    else:
        average_data += data
        average_data_squared += data ** 2

print(number_of_files)
average_data = average_data / number_of_files
quantile_position_var = average_data_squared[:, 2] / number_of_files - average_data[:, 2]**2
average_var = average_data[:, 1]
distance = average_data[:, 0]

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.plot(distance, average_var, c='b', label='Sampling')
ax.plot(distance, quantile_position_var, c='r', label='Env')
ax.set_ylim([10**-3, 10**12])
ax.legend()
fig.savefig("Var.pdf")