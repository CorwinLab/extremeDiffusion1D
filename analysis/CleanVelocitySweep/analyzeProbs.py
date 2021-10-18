import sys

sys.path.append("../../src")
from databases import VelocityDatabase
import glob
import numpy as np
import npquad
from matplotlib import pyplot as plt
import os

files = glob.glob("/home/jacob/Desktop/corwinLabMount/CleanData/VelocitySweep/*.txt")
files=files[:5000] # I'm not sure if all the datasets go out to the final time
db = VelocityDatabase(files)

run_again = True

if not os.path.exists("./Mean.txt") or not os.path.exists("./Vars.txt") or run_again:
    print("Calculating Mean and Variance")
    db.calculateMeanVar(verbose=True)
    np.savetxt("Mean.txt", db.mean)
    np.savetxt("Vars.txt", db.var)
else:
    db.loadMean("Mean.txt")
    db.loadVar("Vars.txt")
print("Number of files:", len(files))

db.plotMeans("./figures/Means")
db.plotVars("./figures/Vars")
db.plotDistribution("./figures/Histograms", verbose=True)
db.plotAllVars("./figures/")
db.plotResidual("./figures/")
