import numpy as np 
import pandas as pd
import npquad
import os
import glob
import sys 
from matplotlib import pyplot as plt

beta_dir = "/home/jacob/Desktop/corwinLabMount/CleanData/CDFBetaSweep/0.01/Q*.txt"
files = glob.glob(beta_dir)
for f in files: 
    data = pd.read_csv(f)
    var = data['100']
    print(var.values[-1])