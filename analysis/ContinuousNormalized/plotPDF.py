import numpy as np
import glob 
from matplotlib import pyplot as plt

dir = '/home/jacob/Desktop/corwinLabMount/CleanData/ContinuousNormalized/P*.txt'
files = glob.glob(dir)

data = np.loadtxt(files[0])
for f in files[1:]: 
    data = np.append(data, np.loadtxt(f))

fig, ax = plt.subplots()
ax.hist(data, density=True, bins=75)
ax.set_yscale("log")
ax.set_xlabel("Position")
ax.set_ylabel("Probability Density")
fig.savefig("ParticlePositions.png", bbox_inches='tight')