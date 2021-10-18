import numpy as np
import npquad
from matplotlib import pyplot as plt
import sys
sys.path.append("../../src")
import theory as th
from quadMath import prettifyQuad

N = np.quad("1e100")
logN = np.log(N).astype(np.float64)
discrete_time = np.loadtxt("./Times.txt")
discrete_var = np.loadtxt("MaxVar.txt")

cdf_time = np.loadtxt("../CDFVar100/Times.txt")
cdf_quantile_var = np.loadtxt("../CDFVar100/QuantileVar.txt")
cdf_discrete_var = np.loadtxt("../CDFVar100/DiscreteVariance.txt")

short_time = th.theoreticalNthQuartVar(N, discrete_time)
long_time = th.theoreticalNthQuartVarLargeTimes(N, discrete_time)

fig, ax = plt.subplots()
ax.plot(discrete_time / logN, discrete_var)
ax.plot(cdf_time / logN, cdf_quantile_var)
ax.plot(cdf_time / logN, cdf_discrete_var)
ax.plot(discrete_time / logN, short_time)
ax.plot(discrete_time / logN, long_time)
ax.plot(discrete_time / logN, discrete_time / logN)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim([10**-4, 10**5])
ax.set_xlim([10**-2, 10**5])
ax.set_xlabel("Time / lnN")
ax.set_ylabel("Variance")
ax.set_title(f"N={prettifyQuad(N)}")
ax.grid(True)
fig.savefig("DiscreteWithCDF.png")
