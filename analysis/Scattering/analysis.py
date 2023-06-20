import numpy as np
from matplotlib import pyplot as plt 
import glob 

folder = '/home/jacob/Desktop/talapasMount/JacobData/Scattering/Q*.txt'
files = glob.glob(folder)
maxTime = 100000-1
data = np.loadtxt(files[0], skiprows=1, delimiter=',')
first_moment = data[:, 1]
second_moment = data[:, 1] ** 2

rerun = False
if rerun: 
    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_yscale("log")
    max_at_100 = []
    max_at_end = []
    num = 1
    for i, f in enumerate(files[1:]): 
        data = np.loadtxt(f, skiprows=1, delimiter=',')
        if data[-1, 0] < maxTime:
            continue 
        max_at_end.append(data[-1, 1])
        max_at_100.append(data[100, 1])
        first_moment += data[:, 1]
        second_moment += data[:, 1] ** 2
        num += 1
        ax.plot(data[:, 0], data[:, 1])
        print(i / len(files))

    mean = first_moment / num
    var = second_moment / num - mean**2
    time = data[:, 0]
    ax.set_xlabel("x")
    ax.set_ylabel("Max")
    fig.savefig("Positions.png", bbox_inches='tight')
    np.savetxt("Mean.txt", mean)
    np.savetxt("Var.txt", var)
    np.savetxt("Time.txt", data[:, 0])
    np.savetxt("Quantile100.txt", max_at_100)
    np.savetxt("QuantileEnd.txt", max_at_end)
else:
    mean = np.loadtxt("Mean.txt")
    var = np.loadtxt("Var.txt")
    time = np.loadtxt("Time.txt")
    quantile_end = np.loadtxt("QuantileEnd.txt")
    quantile_100 = np.loadtxt("Quantile100.txt")

N = 1e10
logN = np.log(N)

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t / \log(N)$")
ax.set_ylabel("Mean(Env)")
ax.plot(time, mean)
fig.savefig("Mean.pdf", bbox_inches='tight')

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t / \log(N)$")
ax.set_ylabel("Var(Env)")
ax.plot(time, var)
fig.savefig("Var.pdf", bbox_inches='tight')

fig, ax = plt.subplots()
ax.hist(quantile_100 - np.mean(quantile_100), bins=500)
fig.savefig("Quantile100.png")

fig, ax = plt.subplots()
ax.hist(quantile_end - np.mean(quantile_end), bins=500)
fig.savefig("QuantileEnd.png")