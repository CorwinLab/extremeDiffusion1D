import sys

sys.path.append("../../src")
import numpy as np
import npquad
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from databases import QuartileDatabase
import glob
import os
from quadMath import prettifyQuad, logarange
import theory as th
from fileIO import loadArrayQuad

files = glob.glob("/home/jacob/Desktop/corwinLabMount/CleanData/MaxPart/Q*.txt")
print("Number of files: ", len(files))
'''
files = glob.glob("/home/jacob/Desktop/corwinLabMount/CleanData/MaxPart/O*.txt")
for f in files:
    occ = np.loadtxt(f, delimiter=",")
    print(np.where(occ != 0))
    print("{:e}".format(np.sum(occ)))
'''

db = QuartileDatabase(files, nParticles=np.quad("1e10"))

run_again = True
if not os.path.exists("./Mean.txt") or not os.path.exists("./Var.txt") or run_again:
    db.calculateMeanVar(verbose=False)
    np.savetxt("Mean.txt", db.mean)
    np.savetxt("Var.txt", db.var)
else:
    db.loadMean("Mean.txt")
    db.loadVar("Var.txt")

print("Maximum Time:", max(db.time))

db.plotMeans(save_dir="./figures/Means/", verbose=True)
db.plotVars(save_dir="./figures/Vars/", verbose=True)
db.plotMaxMean(save_dir="./figures/Means/")
db.plotMaxVar(save_dir="./figures/Vars")
