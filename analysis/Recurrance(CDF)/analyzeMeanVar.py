import numpy as np
import glob
import sys
sys.path.append("../../src")
from databases import CDFQuartileDatabase

files = glob.glob("/home/jacob/Desktop/corwinLabMount/CleanData/Recurrence/Q*.txt")

max_files = []
for f in files:
    time = np.loadtxt(f, delimiter=',', skiprows=1, usecols=0)
    if max(time) == 3_000_000:
        max_files.append(f)

files = max_files

db = CDFQuartileDatabase(files)

run_again = True
if run_again:
    db.calculateMeanVar()
    np.savetxt("Mean.txt", db.mean)
    np.savetxt("Var.txt", db.var)
else:
    db.loadVar('Var.txt')
    db.loadMean('Mean.txt')

db.plotMeans('./figures/Means')
db.plotVars('./figures/Vars')
