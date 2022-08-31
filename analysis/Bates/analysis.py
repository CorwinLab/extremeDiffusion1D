import numpy as np
from matplotlib import pyplot as plt
import sys 
import os
sys.path.append("../../dataAnalysis")
from overalldatabase import Database

print("Beta Variance:", 1/84)
# Test if random numbers are actually bates distributed
file = '/home/jacob/Desktop/corwinLabMount/CleanData/Bates/RandomNums1.txt'
data = np.loadtxt(file)
fig, ax = plt.subplots()
ax.hist(data, bins=50)
print("Bates Variance:", np.var(data))
fig.savefig("RandomNumsBates.png", bbox_inches='tight')

# Test if random numbers are uniform distributed 
file = '/home/jacob/Desktop/corwinLabMount/CleanData/Uniform/RandomNums1.txt'
data = np.loadtxt(file)
fig, ax = plt.subplots()
ax.hist(data, bins=50)
ax.set_xlim([0, 1])
print("Uniform Variance:", np.var(data))
fig.savefig("RandomNumsUniform.png", bbox_inches='tight')

# Test if random numbers are triangular distributed
file = '/home/jacob/Desktop/corwinLabMount/CleanData/Triang/RandomNums1.txt'
data = np.loadtxt(file)
fig, ax = plt.subplots()
ax.hist(data, bins=50)
ax.set_xlim([0, 1])
print("Triang Variance:", np.var(data))
fig.savefig("RandomNumsTriang.png", bbox_inches='tight')

# Plot the variance of each distribution
# Get Bates data
logN = np.log(1e24)
alpha = 0.75
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t/log(N)$")
ax.set_ylabel(r"$\mathrm{Var}(\mathrm{Env}_N^t)$")
ax.set_xlim([0.1, 5 * 10**2])

db = Database()
path = "/home/jacob/Desktop/corwinLabMount/CleanData/Bates/"
db.add_directory(path, dir_type="Gumbel")
#db.calculateMeanVar(path, verbose=True, maxTime=27631)
N = 24
cdf_df, max_df = db.getMeanVarN(N)
ax.plot(cdf_df['time'] / logN, cdf_df['Var Quantile'], label=r'$\mathrm{Bates}(n=7)$', alpha=alpha)

# Get beta=10 data
path = "/home/jacob/Desktop/corwinLabMount/CleanData/CDFBetaSweep/10/"
db = Database()
db.add_directory(path, dir_type="Gumbel")
cdf_df, max_df = db.getMeanVarN(N)
ax.plot(cdf_df['time'] / logN, cdf_df['Var Quantile'], label=r'$\mathrm{Beta}(\alpha=\beta=10)$', alpha=alpha)

# Get beta=0.1 data
path = "/home/jacob/Desktop/corwinLabMount/CleanData/CDFBetaSweep/0.1/"
db = Database()
db.add_directory(path, dir_type="Gumbel")
cdf_df, max_df = db.getMeanVarN(N)
ax.plot(cdf_df['time'] / logN, cdf_df['Var Quantile'], label=r'$\mathrm{Beta}(\alpha=\beta=0.1)$', alpha=alpha)

# Get beta=1 data
path = "/home/jacob/Desktop/corwinLabMount/CleanData/CDFBetaSweep/1/"
db = Database()
db.add_directory(path, dir_type="Gumbel")
cdf_df, max_df = db.getMeanVarN(N)
ax.plot(cdf_df['time'] / logN, cdf_df['Var Quantile'], label=r'$\mathrm{Beta}(\alpha=\beta=1)$', alpha=alpha)

# Get Uniform distributed
db = Database()
path = "/home/jacob/Desktop/corwinLabMount/CleanData/Uniform/"
db.add_directory(path, dir_type="Gumbel")
#db.calculateMeanVar(path, verbose=True, maxTime=27631)
cdf_df, max_df = db.getMeanVarN(N)
ax.plot(cdf_df['time'] / logN, cdf_df['Var Quantile'], label=r'$\mathrm{Uniform}(\frac{1}{2}(1 \pm 1/\sqrt{7}))$', alpha=alpha)

# Get triangle distributed
db = Database()
path = "/home/jacob/Desktop/corwinLabMount/CleanData/Triang/"
db.add_directory(path, dir_type="Gumbel")
db.calculateMeanVar(path, verbose=True, maxTime=27631)
cdf_df, max_df = db.getMeanVarN(N)
ax.plot(cdf_df['time'] / logN, cdf_df['Var Quantile'], label=r'$\mathrm{Traing}(1/2 \pm 1/\sqrt{14}, 1/2)$', alpha=alpha)

ax.legend(loc='upper left')
fig.savefig("Variance.pdf", bbox_inches='tight')