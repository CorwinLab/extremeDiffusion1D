import glob
import numpy as np
import sys
sys.path.append("../../src/")
from fileIO import loadArrayQuad

files = glob.glob("/home/jacob/Desktop/corwinLabMount/CleanData/QSweep/O*.txt")

print("Number of files:", len(files))
for f in files:
    occ = loadArrayQuad(f, skiprows=0, shape=300001)
    print(sum(occ))
    maxIdx = max(np.argwhere(occ))
    print(occ[maxIdx]) # These are < 1 so it for sure isn't discrete! 
