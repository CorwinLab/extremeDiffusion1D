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

files = glob.glob(
    "/home/jacob/Desktop/corwinLabMount/CleanData/QSweep/Q*.txt"
)

print("Number of files: ", len(files))

db = QuartileDatabase(files)

run_again = False
if not os.path.exists("./Mean.txt") or not os.path.exists("./Var.txt") or run_again:
    db.calculateMeanVar(verbose=True)
    np.savetxt("Mean.txt", db.mean)
    np.savetxt("Var.txt", db.var)
else:
    db.loadMean("Mean.txt")
    db.loadVar("Var.txt")

print("Maximum Time:", max(db.time))

db.plotMeans(save_dir='./figures/Means/', verbose=True)
db.plotVars(save_dir='./figures/Vars/', verbose=True)
db.plotVarsEvolve(save_dir='./figures/Vars/')
