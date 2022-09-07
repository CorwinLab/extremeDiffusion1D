import numpy as np
from matplotlib import pyplot as plt
import sys 
import os
sys.path.append("../../dataAnalysis")
from overalldatabase import Database

# Plot the variance of each distribution
# Get Bates data
N = 24
logN = np.log(float(f"1e{N}"))
alpha = 0.6
lw = 1
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t/log(N)$")
ax.set_ylabel(r"$\mathrm{Mean}(\mathrm{Env}_N^t)$")
ax.set_xlim([1/logN, 5 * 10 **2  * np.log(1e24)/logN])

db = Database()
path = "/home/jacob/Desktop/corwinLabMount/CleanData/Bates/"
db.add_directory(path, dir_type="Gumbel")
#db.calculateMeanVar(path, verbose=True, maxTime=27631)
cdf_df, max_df = db.getMeanVarN(N)
ax.plot(cdf_df['time'] / logN, cdf_df['Mean Quantile'], alpha=alpha, lw=lw, label=r'$\mathrm{Bates}(\beta=10)$')

# Get beta=10 data
path = "/home/jacob/Desktop/corwinLabMount/CleanData/CDFBetaSweep/10/"
db = Database()
db.add_directory(path, dir_type="Gumbel")
cdf_df, max_df = db.getMeanVarN(N)
ax.plot(cdf_df['time'] / logN, cdf_df['Mean Quantile'], label=r'$\mathrm{Beta}(\alpha=\beta=10)$', alpha=alpha, lw=lw)

# Get beta=0.1 data
path = "/home/jacob/Desktop/corwinLabMount/CleanData/CDFBetaSweep/0.1/"
db = Database()
db.add_directory(path, dir_type="Gumbel")
cdf_df, max_df = db.getMeanVarN(N)
ax.plot(cdf_df['time'] / logN, cdf_df['Mean Quantile'], label=r'$\mathrm{Beta}(\alpha=\beta=0.1)$', alpha=alpha, lw=lw)

# Get beta=1 data
path = "/home/jacob/Desktop/corwinLabMount/CleanData/CDFBetaSweep/1/"
db = Database()
db.add_directory(path, dir_type="Gumbel")
cdf_df, max_df = db.getMeanVarN(N)
ax.plot(cdf_df['time'] / logN, cdf_df['Mean Quantile'], label=r'$\mathrm{Beta}(\alpha=\beta=1)$', alpha=alpha, lw=lw)

# Get Uniform distributed
db = Database()
path = "/home/jacob/Desktop/corwinLabMount/CleanData/Uniform/"
db.add_directory(path, dir_type="Gumbel")
#db.calculateMeanVar(path, verbose=True, maxTime=27631)
cdf_df, max_df = db.getMeanVarN(N)
ax.plot(cdf_df['time'] / logN, cdf_df['Mean Quantile'], alpha=alpha, lw=lw, label=r'$\mathrm{Uniform}(\beta=10)$' )

# Get triangle distributed
db = Database()
path = "/home/jacob/Desktop/corwinLabMount/CleanData/Triang/"
db.add_directory(path, dir_type="Gumbel")
#db.calculateMeanVar(path, verbose=True, maxTime=27631)
cdf_df, max_df = db.getMeanVarN(N)
ax.plot(cdf_df['time'] / logN, cdf_df['Mean Quantile'], alpha=alpha, lw=lw, label=r'$\mathrm{Traing}(\beta=10)$')

# Get triangle distributed
db = Database()
path = "/home/jacob/Desktop/corwinLabMount/CleanData/Quadratic/"
db.add_directory(path, dir_type="Gumbel")
#db.calculateMeanVar(path, verbose=True, maxTime=27631)
cdf_df, max_df = db.getMeanVarN(N)
ax.plot(cdf_df['time'] / logN, cdf_df['Mean Quantile'], alpha=alpha, lw=lw, label=r'$\mathrm{Quadratic}(\beta=10)$')

db = Database()
path = "/home/jacob/Desktop/corwinLabMount/CleanData/QuadraticBeta1/"
db.add_directory(path, dir_type="Gumbel")
#db.calculateMeanVar(path, verbose=True, maxTime=55262)
cdf_df, max_df = db.getMeanVarN(N)
ax.plot(cdf_df['time'] / logN, cdf_df['Mean Quantile'], alpha=alpha, lw=lw, label=r'$\mathrm{Quadratic}(\beta=1)$')

db = Database()
path = "/home/jacob/Desktop/corwinLabMount/CleanData/DeltaBeta01/"
db.add_directory(path, dir_type="Gumbel")
#db.calculateMeanVar(path, verbose=True, maxTime=55262)
cdf_df, max_df = db.getMeanVarN(N)
ax.plot(cdf_df['time'] / logN, cdf_df['Mean Quantile'], alpha=alpha, lw=lw, label=r'$Deltas (\beta=0.1)$')

ax.legend(loc='upper left', fontsize=8)
fig.savefig(f"Mean{N}.pdf", bbox_inches='tight')