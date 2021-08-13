import glob
import os
import sys
sys.path.append("../../src/")
from databases import QuartileDatabase
import numpy as np
import npquad

files = glob.glob("/home/jacob/Desktop/corwinLabMount/Data/1.0/QuartileSmall/Q*.txt")
print("Number of files:", len(files))

db = QuartileDatabase(files)
db.calculateMeanVar(verbose=True)
np.savetxt("Mean.txt", db.mean)
np.savetxt("Var.txt", db.var)
db.plotMeans()
db.plotVars()
