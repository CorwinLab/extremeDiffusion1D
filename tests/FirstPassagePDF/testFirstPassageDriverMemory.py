import numpy as np
import npquad 
import sys 
sys.path.append("../../src")
from libDiffusion import FirstPassageDriver
import os, psutil

N = 1e24 
distances = np.unique(np.geomspace(10, 500 * np.log(N), 500).astype(int))
beta = 1

pdf = FirstPassageDriver(beta, distances)
process = psutil.Process(os.getpid())
print(process.memory_info().rss * 1e-6)