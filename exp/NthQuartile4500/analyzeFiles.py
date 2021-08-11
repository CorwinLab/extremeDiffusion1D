import glob
import numpy as np
import npquad
import sys
sys.path.append("../../src")
sys.path.append("../../cDiffusion")
from pydiffusion import loadArrayQuad
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

files = glob.glob("/home/jhass2/Data/1.0/QuartileTotal/Q*.txt")
print(len(files))

max_time = []
for f in files: 
    data = np.loadtxt(f, skiprows=1, delimiter=',')
    max_time.append(data[-1, 0])
    print(f, data[-1, 0])

fig, ax = plt.subplots()
ax.hist(max_time)
ax.set_xscale('log')
fig.savefig("times.png")

