import numpy as np
import npquad
import glob
import pandas as pd
from matplotlib import pyplot as plt

dir = "/home/jacob/Desktop/talapasMount/JacobData/LongFirstPassCDF2/F*.txt"
#dir = '../../Data2/F*.txt'
files = glob.glob(dir)

average_data = None
average_data_squared = None
number_of_files = 0
max_dist = 1000
for f in files:
    data = pd.read_csv(f, delimiter=',') # columns are position, quantile, variance
    
    if max(data['position']) < max_dist: 
        continue
    data.sort_values(by=['position'])
    data = data[data['position'] <= max_dist].values
    number_of_files += 1
    if average_data is None:
        average_data = data
        average_data_squared = data ** 2
    else:
        try:
            average_data += data
            average_data_squared += data ** 2
        except:
            print("something went wrong")
            number_of_files -= 1
            continue

print(number_of_files)
average_data = average_data / number_of_files
quantile_position_var = average_data_squared[:, 1] / number_of_files - average_data[:, 1]**2
average_var = average_data[:, 2]
distance = average_data[:, 0]

N = np.quad("1e2")
logN = np.log(N).astype(float)

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim([10**-3, 10**10])
ax.plot(distance / logN, quantile_position_var, label='Env')
ax.plot(distance / logN, average_var, label='Sam')
ax.legend()
fig.savefig("Var.pdf", bbox_inches='tight')