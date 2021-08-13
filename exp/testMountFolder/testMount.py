import glob
import os
import sys

sys.path.append("../../src/")
from databases import QuartileDatabase
import numpy as np
import npquad

files = glob.glob("/home/jacob/Desktop/corwinLabMount/Data/1.0/QuartileLarge/Q*.txt")
print("Number of files:", len(files))

db = QuartileDatabase(files)
db.calculateMeanVar(verbose=True)
np.savetxt("MeanLarge.txt", db.mean)
np.savetxt("VarLarge.txt", db.var)
db.plotMeans()
db.plotVars()
