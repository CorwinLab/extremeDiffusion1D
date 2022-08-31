import numpy as np
from matplotlib import pyplot as plt
import sys 
import os
sys.path.append("../../dataAnalysis")
from overalldatabase import Database

# Test if random numbers are actually bates distributed
file = '/home/jacob/Desktop/corwinLabMount/CleanData/Bates/RandomNums1.txt'
data = np.loadtxt(file)
fig, ax = plt.subplots()
ax.hist(data, bins=50)
fig.savefig("RandomNums.png", bbox_inches='tight')

db = Database()
path = "/home/jacob/Desktop/corwinLabMount/CleanData/Bates/"
db.add_directory(path, dir_type="Gumbel")
db.calculateMeanVar(path, verbose=True, maxTime=27631)
N = 24
cdf_df, max_df = db.getMeanVarN(N)

logN = np.log(1e24)
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t/log(N)$")
ax.set_ylabel(r"$\mathrm{Var}(\mathrm{Env}_N^t)$")
ax.set_xlim([0.5, 5 * 10**2])
ax.plot(cdf_df['time'] / logN, cdf_df['Var Quantile'], label=r'$\mathrm{Bates}(n=7)$')

# Get beta=10 data
path = "/home/jacob/Desktop/corwinLabMount/CleanData/CDFBetaSweep/10/"
db = Database()
db.add_directory(path, dir_type="Gumbel")
cdf_df, max_df = db.getMeanVarN(N)

ax.plot(cdf_df['time'] / logN, cdf_df['Var Quantile'], label=r'$\mathrm{Beta}(\alpha=\beta=10)$')
ax.legend(loc='upper left')
fig.savefig("Variance.pdf", bbox_inches='tight')