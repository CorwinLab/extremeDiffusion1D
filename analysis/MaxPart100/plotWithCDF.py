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

exclude_times = np.where(cdf_time / logN <= 2)[0][-1]
cdf_discrete_var[0:exclude_times] = 0

Einstein100 = np.loadtxt(
    "/home/jacob/Desktop/corwinLabMount/CleanData/Einstein100/Quartiles0.txt",
    delimiter=",",
    skiprows=1,
)
time = Einstein100[:, 0]
quantile = Einstein100[:, 1]
variance = Einstein100[:, 2]
exlcude_times = np.where(time / logN <= 2)[0][-1]
variance[0:exclude_times] = 0

fig, ax = plt.subplots()
ax.plot(discrete_time / logN, discrete_var, "tab:orange", alpha=0.8)
ax.plot(cdf_time / logN, cdf_quantile_var + cdf_discrete_var, "tab:red")
ax.plot(discrete_time / logN, short_time, "tab:green")
ax.plot(discrete_time / logN, long_time, "tab:orange")
ax.plot(discrete_time / logN, discrete_time / logN, "tab:purple")
ax.plot(time / logN, variance, "tab:olive", ls="--", alpha=0.7)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim([10 ** -4, 10 ** 5])
ax.set_xlim([10 ** -2, 10 ** 5])
ax.set_xlabel("Time / lnN")
ax.set_ylabel("Variance")
ax.set_title(f"N={prettifyQuad(N)}")
ax.grid(True)
fig.savefig("FinalPlotCDF.png")

fig, ax = plt.subplots()
ax.plot(discrete_time / logN, discrete_var, "tab:orange")
ax.plot(cdf_time / logN, cdf_discrete_var, "tab:blue")
ax.plot(discrete_time / logN, short_time + discrete_time / logN, "tab:green")
ax.plot(discrete_time / logN, discrete_time / logN, "tab:purple")
ax.plot(time / logN, variance, "tab:olive", ls="--", alpha=0.7)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim([10 ** -4, 10 ** 5])
ax.set_xlim([10 ** -2, 10 ** 5])
ax.set_xlabel("Time / lnN")
ax.set_ylabel("Variance")
ax.set_title(f"N={prettifyQuad(N)}")
ax.grid(True)
fig.savefig("DiscreteCDF.png")
