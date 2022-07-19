import numpy as np
import npquad 
from matplotlib import pyplot as plt
import glob 

dir = "/home/jacob/Desktop/talapasMount/JacobData/BetaCDF/0.01/Q*.txt"
files = glob.glob(dir)

mean = np.zeros(shape=(4270, 11))
count = 0
for f in files: 
    data = np.loadtxt(f, skiprows=1, delimiter=',')
    if data[-1,0] != 276310.0: # this is the maximum time
        continue
    mean+=data
    count+=1

mean = mean / count

N_exp = [2, 7, 24, 85, 300]

fig, ax = plt.subplots()
for i in range(0, 5):
    ax.plot(mean[:, 0], mean[:, i+1], label=N_exp[i])
ax.set_xscale("log")
ax.set_xlim([1, 10000])
ax.set_ylim([1, 400])
ax.legend()
fig.savefig("Mean.png")