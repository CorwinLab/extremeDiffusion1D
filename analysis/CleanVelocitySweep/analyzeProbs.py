import sys
sys.path.append("../../src")
from databases import VelocityDatabase
import glob
import numpy as np
import npquad
from matplotlib import pyplot as plt
import os

files = glob.glob("/home/jacob/Desktop/corwinLabMount/CleanData/VelocitySweep/*.txt")
files = files[:5000]
print("Number of files:", len(files))

db = VelocityDatabase(files)

run_again = False

if not os.path.exists("./Mean.txt") or not os.path.exists("./Vars.txt") or run_again:
    print("Calculating Mean and Variance")
    db.calculateMeanVar(verbose=True)
    np.savetxt("Mean.txt", db.mean)
    np.savetxt("Vars.txt", db.var)
else:
    db.loadMean("Mean.txt")
    db.loadVar("Vars.txt")

db.plotMeans("./figures/Means")
db.plotVars("./figures/Vars")
db.plotDistribution("./figures/Histograms", verbose=True, load_file='FinalTime.txt')
