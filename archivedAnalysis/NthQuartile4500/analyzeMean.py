import glob
import numpy as np
import npquad
import sys

sys.path.append("../../src")
sys.path.append("../../cDiffusion")
from databases import QuartileDatabase

files = glob.glob("/home/jhass2/Data/1.0/QuartileTotal/Q*.txt")
print("Files Found:", len(files))

full_files = []
time = 0
for f in files:
    data = np.loadtxt(f, skiprows=1, delimiter=",")
    max_time = data[-1, 0]
    if max_time > time:
        full_files = [f]
        time = max_time
    elif max_time == time:
        full_files.append(f)
    else:
        continue

db = QuartileDatabase(full_files)
db.calculateMeanVar()
