import os
import sys 
from matplotlib import pyplot as plt
from scipy.special import digamma

sys.path.append("../../dataAnalysis")
from overalldatabase import Database

db = Database()
beta_dir = "/home/jacob/Desktop/corwinLabMount/CleanData/CDFBetaSweep/"
dirs = os.listdir(beta_dir)
run_again = False
if run_again:
    for dir in dirs:
        path = os.path.join(beta_dir, dir)
        beta = float(dir.split("/")[-1])
        db.add_directory(path, dir_type="Gumbel")
        db.calculateMeanVar(path, verbose=True, maxTime=276310)

max_dir = "/home/jacob/Desktop/talapasMount/JacobData/BetaSweepPaper"
dirs = os.listdir(max_dir)
run_again = False
if run_again:
    for dir in dirs:
        path = os.path.join(max_dir, dir)
        beta = float(dir.split("/")[-1])
        db.add_directory(path, dir_type="Max")
        db.calculateMeanVar(path, verbose=True, maxTime=276310)