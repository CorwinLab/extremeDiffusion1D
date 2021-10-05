from matplotlib import pyplot as plt
import numpy as np
import glob
import sys
sys.path.append("../../src")
import theory as th

files = glob.glob("/home/jacob/Desktop/corwinLabMount/CleanData/CDFVarFixed/Quartiles*.txt")
finished_files = []
nParticles = 1000000

for f in files:
    try:
        times = np.loadtxt(f, delimiter=',', skiprows=1, usecols=0)
    except:
        continue
    if max(times) == 100_000:
        finished_files.append(f)
        len_times = len(times)

print(len(finished_files))
discrete_var = np.zeros(len_times)
quartile_sum = np.zeros(len_times)
quartile_squared_sum = np.zeros(len_times)

for f in finished_files:
    data = np.loadtxt(f, delimiter=',', skiprows=1)
    time = data[:, 0]
    quartile_sum += data[:, 1]
    quartile_squared_sum += data[:, 1]**2
    discrete_var += data[:, 2]

quartile_var = quartile_squared_sum / len(finished_files) - (quartile_sum / len(finished_files)) ** 2
discrete_var = discrete_var / len(finished_files)

theoryVar = th.theoreticalNthQuartVar(nParticles, time)
theoryVarLongTimes = th.theoreticalNthQuartVarLargeTimes(nParticles, time)
np.savetxt("QuantileVar.txt", quartile_var)
np.savetxt("DiscreteVariance.txt", discrete_var)
np.savetxt("Times.txt", time)
fig, ax = plt.subplots()
ax.plot(time / np.log(nParticles), quartile_var, label="CDF Quantile")
ax.plot(time / np.log(nParticles), discrete_var, label="Max Particle")
ax.plot(time / np.log(nParticles), time / np.log(nParticles), label='Linear')
ax.plot(time / np.log(nParticles), theoryVar, label=th.NthQuartVarStr)
ax.plot(time / np.log(nParticles), theoryVarLongTimes, label=th.NthQuartVarStrLargeTimes)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Time / lnN")
ax.set_ylabel("Variance")
ax.set_title(f"N={nParticles}")
ax.grid(True)
ax.set_ylim([10**-1, max(discrete_var)])
ax.legend()
fig.savefig("Variance.png")
