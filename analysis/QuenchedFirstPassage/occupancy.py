import sys
sys.path.append("../../src")

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import glob
import os
import pandas as pd

def last_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)

def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

file = "/home/jacob/Desktop/talapasMount/JacobData/QuenchedFirstPassageDiscrete/FinalOccupancy500.txt"
times = np.loadtxt("/home/jacob/Desktop/talapasMount/JacobData/QuenchedFirstPassageDiscrete/Quartiles500.txt", skiprows=1, delimiter=',')
time = times[-1, 1]
maxTime = 3000000
xvals = np.arange(0, maxTime+1) - time / 2
occupancy = np.loadtxt(file, delimiter=',')
last_idx = last_nonzero(occupancy, 0)
first_idx = first_nonzero(occupancy, 0)
occupancy = occupancy[first_idx-1:last_idx+2]
xvals = xvals[first_idx-1:last_idx+2]
fig, ax = plt.subplots()
ax.scatter(xvals, occupancy)
ax.set_yscale("log")
ax.set_ylim([1, 1e24])
fig.savefig("Occ.png")