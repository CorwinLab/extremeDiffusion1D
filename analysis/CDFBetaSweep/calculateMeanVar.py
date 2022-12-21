import os
import sys 
from matplotlib import pyplot as plt
from scipy.special import digamma

sys.path.append("../../dataAnalysis")
from overalldatabase import Database

db = Database()
beta_dir = "/home/jacob/Desktop/corwinLabMount/CleanData/CDFBetaSweep/"
dirs = os.listdir(beta_dir)
for dir in dirs:
    path = os.path.join(beta_dir, dir)
    beta = float(dir.split("/")[-1])
    db.add_directory(path, dir_type="Gumbel")
    db.calculateMeanVar(path, verbose=True, maxTime=276310)