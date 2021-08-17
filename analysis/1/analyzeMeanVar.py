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

files = glob.glob("/home/jacob/Desktop/corwinLabMount/Data/1.0/1/Q*.txt")
print("Number of files: ", len(files))

Ns = np.geomspace(1e10, 1e50, 9)
db = QuartileDatabase(files, delimiter=" ", skiprows=0)
db.setNs(Ns)

run_again = False
if not os.path.exists("./Mean.txt") or not os.path.exists("./Var.txt") or run_again:
    db.calculateMeanVar(verbose=True)
    np.savetxt("Mean.txt", db.mean)
    np.savetxt("Var.txt", db.var)
else:
    db.loadMean("Mean.txt")
    db.loadVar("Var.txt")

print("Maximum time: ", max(db.time))

db.plotMeans(save_dir='./figures/')
db.plotVars(save_dir='./figures/')
