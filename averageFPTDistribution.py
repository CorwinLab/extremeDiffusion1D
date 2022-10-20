import numpy as np
import glob 
from matplotlib import pyplot as plt
'''
files = glob.glob("./FPTCDF/CDF*.txt")
maxTime = 3_900_000
number_of_files = 0
sum = np.zeros(shape=maxTime)
for f in files[:10]: 
    data = np.loadtxt(f)
    if len(data) < maxTime:
        print("Not large enough")
        continue
    data = data[:maxTime]
    sum += data
    number_of_files += 1
    print(f)
mean = sum / number_of_files
np.savetxt("./FPTCDF/Mean.txt", mean)
'''

'''Plot the results - it's not that interesting
mean = np.loadtxt("./FPTCDF/Mean.txt")
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim([10**-200, 1.1])
ax.plot(np.arange(1, len(mean[np.nonzero(mean)])+1), mean[np.nonzero(mean)])
fig.savefig("FPT.png")
'''