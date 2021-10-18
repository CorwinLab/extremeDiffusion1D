from matplotlib import pyplot as plt
import numpy as np
import glob
import sys
sys.path.append("../../src")
import theory as th

files = glob.glob("/home/jacob/Desktop/corwinLabMount/CleanData/CDFVar100/Quartiles*.txt")
nParticles = np.quad("1e100")

maxTime = 2000000
discrete_var = None
quartile_sum = None
quartile_squared_sum = None
number_of_files = 0
plot_time = []

for f in files:
    data = np.loadtxt(f, delimiter=',', skiprows=1)
    time = data[:, 0]

    if max(time) < maxTime:
        continue
    time = time[time<=maxTime]
    maxIdx = len(time)
    plot_time = time

    if discrete_var is None:
        discrete_var = np.zeros(maxIdx)
        quartile_sum = np.zeros(maxIdx)
        quartile_squared_sum = np.zeros(maxIdx)

    quartile_sum += data[:maxIdx, 1]
    quartile_squared_sum += data[:maxIdx, 1]**2
    discrete_var += data[:maxIdx, 2]
    number_of_files += 1

print(f"Number of files taken: {number_of_files}")
quartile_var = quartile_squared_sum / number_of_files - (quartile_sum / number_of_files) ** 2
discrete_var = discrete_var / number_of_files

theoryVar = th.theoreticalNthQuartVar(nParticles, plot_time)
theoryVarLongTimes = th.theoreticalNthQuartVarLargeTimes(nParticles, plot_time)
np.savetxt("QuantileVar.txt", quartile_var)
np.savetxt("DiscreteVariance.txt", discrete_var)
np.savetxt("Times.txt", plot_time)
fig, ax = plt.subplots()
ax.plot(plot_time / np.log(nParticles), quartile_var, label="CDF Quantile")
ax.plot(plot_time / np.log(nParticles), discrete_var, label="Max Particle")
ax.plot(plot_time / np.log(nParticles), plot_time / np.log(nParticles), label='Linear')
ax.plot(plot_time / np.log(nParticles), theoryVar, label=th.NthQuartVarStr)
ax.plot(plot_time / np.log(nParticles), theoryVarLongTimes, label=th.NthQuartVarStrLargeTimes)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Time / lnN")
ax.set_ylabel("Variance")
ax.set_title(f"N={nParticles}")
ax.grid(True)
ax.set_ylim([10**-1, max(discrete_var)])
ax.legend()
fig.savefig("Variance.png")
