import os
import sys 
from matplotlib import pyplot as plt
from scipy.special import digamma

sys.path.append("../../dataAnalysis")
from overalldatabase import Database

# Calculate beta distribution cdf
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

# Calculate beta distribution maximum
max_dir = "/home/jacob/Desktop/talapasMount/JacobData/BetaSweepPaper"
dirs = os.listdir(max_dir)
run_again = False
if run_again:
    for dir in dirs:
        path = os.path.join(max_dir, dir)
        beta = float(dir.split("/")[-1])
        db.add_directory(path, dir_type="Max")
        db.calculateMeanVar(path, verbose=True, maxTime=276310)

# Calculate bates distribution cdf
bates_dir = "/home/jacob/Desktop/talapasMount/JacobData/Bates"
dirs = os.listdir(bates_dir)
run_again = False
if run_again:
    for dir in dirs:
        path = os.path.join(bates_dir, dir)
        db.add_directory(path, dir_type="Gumbel")
        db.calculateMeanVar(path, verbose=True, maxTime=276310)

# Calculate delta distribution cdf
delta_dir = "/home/jacob/Desktop/talapasMount/JacobData/Delta"
dirs = os.listdir(delta_dir)
run_again = False
if run_again:
    for dir in dirs:
        path = os.path.join(delta_dir, dir)
        db.add_directory(path, dir_type="Gumbel")
        db.calculateMeanVar(path, verbose=True, maxTime=276310)