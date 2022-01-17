from matplotlib import pyplot as plt
import numpy as np
import npquad
import glob
import sys

sys.path.append("../../src")
#from theory import theoreticalNthQuartVar, NthQuartVarStr

files = glob.glob(
    "/home/jacob/Desktop/corwinLabMount/Data/Einstein/1.00e_50/Edges*.txt"
)

mean_sum = None
squared_sum = None
run_again = False
if run_again:
    for f in files:
        data = np.loadtxt(f)
        data = data[1:, :]
        time = np.arange(1, len(data[:, 0]) + 1)
        center = time * 0.5
        max_disp = 2 * (data[:, 1] - center)
        if mean_sum is None:
            mean_sum = np.zeros(len(center))
            squared_sum = np.zeros(len(center))
        mean_sum += max_disp
        squared_sum += max_disp ** 2
        print(f)

    var = squared_sum / len(files) - (mean_sum / len(files)) ** 2
    np.savetxt("Variance.txt", var)
    np.savetxt("Mean.txt", mean_sum / len(files))
    np.savetxt("Time.txt", time)

var = np.loadtxt("Variance.txt")
mean = np.loadtxt("Mean.txt")
time = np.loadtxt("Time.txt")
N = 1e50
logN = np.log2(N)

mean_theory = np.piecewise(time,
                           [time < logN, time >= logN],
                           [lambda t: t, lambda t: np.sqrt(2 * t * np.log(N))])
var_theory = np.piecewise(time,
                          [time < logN, time >= logN],
                          [0, lambda t: np.pi**2 / 12 * t/np.log(N)])

fig, ax = plt.subplots()
ax.set_xlabel(r"$t / \log_2(N)$")
ax.set_ylabel(r"$\sigma^2_{max}(t)$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.plot(time / logN, var, c='k')
ax.plot(time / logN, var_theory, c='r')
ax.set_xlim([0.5, max(time / logN)])
fig.savefig("Variance.png", bbox_inches='tight')

fig, ax = plt.subplots()
ax.set_xlabel(r"$t / \log_2(N)$")
ax.set_ylabel(r"$\bar{X}_{max}(t)$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.plot(time / logN, mean, zorder=0, c='k')
ax.plot(time / logN, mean_theory, '--', c='r')
ax.set_xlim([min(time/logN), max(time/logN)])
ax.set_ylim([1, 10**4])
fig.savefig("Mean.png", bbox_inches='tight')
