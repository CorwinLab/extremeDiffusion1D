import sys
sys.path.append("../../src")
import numpy as np
import npquad
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from databases import QuartileDatabase
import glob

files = glob.glob("/home/jacob/Desktop/corwinLabMount/Data/1.0/QuartileSmall/Q*.txt")
print("Number of files: ", len(files))

db = QuartileDatabase(files)
db.calculateMeanVar(verbose=True)
np.savetxt("Mean.txt", db.mean)
np.savetxt("Var.txt", db.var)
db.plotMeans(save_dir='./figures/')
db.plotVars(save_dir='./figures/')
