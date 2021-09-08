import numpy as np
import glob
import sys
sys.path.append("../../src")
from databases import QuartileDatabase

files = glob.glob("/home/jacob/Desktop/corwinLabMount/CleanData/QSweep/Q*.txt")

db = QuartileDatabase(files)
db.loadMean("Mean.txt")
db.loadVar("Var.txt")
db.plotGumbalDist(np.quad("1e15"), verbose=True)
db.plotGumbalDistOverTime(np.quad("1e15"))
