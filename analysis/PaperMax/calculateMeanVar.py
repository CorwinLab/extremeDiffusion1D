import sys

sys.path.append("../../dataAnalysis")

from overalldatabase import Database
from matplotlib import pyplot as plt
import theory
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import npquad
import json
import os

db = Database()
einstein_dir = "/home/jacob/Desktop/corwinLabMount/CleanData/JacobData/EinsteinPaper/"
directory = "/home/jacob/Desktop/corwinLabMount/CleanData/Paper/Max/"
cdf_path = "/home/jacob/Desktop/corwinLabMount/CleanData/Paper/CDF/"
cdf_path_talapas = "/home/jacob/Desktop/corwinLabMount/CleanData/JacobData/Paper/"
dirs = os.listdir(directory)
for dir in dirs:
    path = os.path.join(directory, dir)
    db.add_directory(path, dir_type="Max")
    N = int(path.split("/")[-1])
    db.calculateMeanVar(path, verbose=True)

e_dirs = os.listdir(einstein_dir)
for dir in e_dirs:
    path = os.path.join(einstein_dir, dir)
    N = int(path.split("/")[-1])
    if N == 300:
        continue
    db.add_directory(path, dir_type="Max")
    # db.calculateMeanVar(path, verbose=True)

db.add_directory(cdf_path, dir_type="Gumbel")
db.add_directory(cdf_path_talapas, dir_type="Gumbel")
# db.calculateMeanVar([cdf_path, cdf_path_talapas], verbose=True, maxTime=3453876)

db1 = db.getBetas(1)
for dir in db.dirs.keys():
    f = open(os.path.join(dir, "analysis.json"), "r")
    x = json.load(f)
    print(dir, " Systems:", x["number_of_systems"])

dbe = db.getBetas(float("inf"))
quantiles = db1.N(dir_type="Max")