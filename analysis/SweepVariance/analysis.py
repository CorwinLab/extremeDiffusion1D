import numpy as np
import npquad
from matplotlib import pyplot as plt
import sys
sys.path.append("../../src")
from databases import CDFVarianceDatabase
import glob
from quadMath import prettifyQuad

files = glob.glob("/home/jacob/Desktop/corwinLabMount/CleanData/SweepVariance/Quartiles*.txt")
db = CDFVarianceDatabase(files)

db.calculateMeanVar(verbose=True, maxTime=2000000)
quantiles = db.quantile_list
np.savetxt("Mean.txt", db.mean)
np.savetxt("Var.txt", db.var)
np.savetxt("GumbelMean.txt", db.gumbelMean)

for i in range(db.gumbelMean.shape[1]):
    N = np.quad(quantiles[i])
    logN = np.log(N).astype(float)
    Nstr = prettifyQuad(N)
    fig, ax = plt.subplots()
    ax.plot(db.time / logN, db.gumbelMean[:, i])
    ax.plot(db.time / logN, db.time / logN, '--')
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Time / lnN")
    ax.set_ylabel("Variance")
    ax.set_title(f"N={Nstr}")
    ax.set_ylim([10**-4, max(db.gumbelMean[:, i])])
    fig.savefig(f"./figures/Var{Nstr}.png")

'''
# Take a look at the old data
files = glob.glob("/home/jacob/Desktop/corwinLabMount/CleanData/CDFVar100/Quartiles*.txt")
nParticles = np.quad("1e100")
Nstr = prettifyQuad(nParticles)

maxTime = 100000
discrete_var = None
quartile_sum = None
quartile_squared_sum = None
number_of_files = 0
plot_time = []

for f in files:
    data = np.loadtxt(f, delimiter=",", skiprows=1)
    time = data[:, 0]

    if max(time) < maxTime:
        continue
    time = time[time <= maxTime]
    maxIdx = len(time)
    plot_time = time

    if discrete_var is None:
        discrete_var = np.zeros(maxIdx)
        quartile_sum = np.zeros(maxIdx)
        quartile_squared_sum = np.zeros(maxIdx)

    quartile_sum += data[:maxIdx, 1]
    quartile_squared_sum += data[:maxIdx, 1] ** 2
    discrete_var += data[:maxIdx, 2]
    number_of_files += 1

print(f"Number of files taken: {number_of_files}")
quartile_var = (
    quartile_squared_sum / number_of_files - (quartile_sum / number_of_files) ** 2
)
discrete_var = discrete_var / number_of_files

logN = np.log(nParticles).astype(float)
fig, ax = plt.subplots()
ax.plot(time / logN, discrete_var)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Time / lnN")
ax.set_ylabel("Variance")
ax.set_title(f"N={Nstr}")
ax.set_ylim([10**-4, max(discrete_var)])
fig.savefig(f"./figures/OldVar{Nstr}.png")
'''
