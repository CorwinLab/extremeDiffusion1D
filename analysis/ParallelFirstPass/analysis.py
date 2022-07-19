import numpy as np
import glob
import matplotlib.pyplot as plt

files = glob.glob('/home/jacob/Desktop/talapasMount/JacobData/ParallelFirstPass/24/F*.txt')

mean_envs = []
var_envs = []
data0 = np.loadtxt(files[0])

sum = np.zeros(data0.shape)
square_sum = np.zeros(data0.shape)

for f in files: 
    data = np.loadtxt(f)
    sum += data 
    square_sum += data**2
    numSystems = data.shape[0]
    mean = np.mean(data, axis=0)
    var = np.var(data, axis=0)

    mean_envs.append(mean)
    var_envs.append(var)

mean_envs = np.array(mean_envs)
var_envs = np.array(var_envs)
numEnvs = var_envs.shape[0]

mean_var_envs = np.mean(var_envs, axis=0)
var_mean_envs = np.var(mean_envs, axis=0)

distances = np.geomspace(1, 2000, 1500, dtype=np.int64)
distances = np.unique(distances)
logN = np.log(1e24)

numSystemsTotal = data0.shape[0] * len(files)
mean_total = np.sum(sum, axis=0) / numSystemsTotal
var_total = np.sum(square_sum, axis=0) / numSystemsTotal - mean_total**2

fig, ax = plt.subplots()
ax.set_xlabel("x / logN")
ax.set_ylabel("Var")
ax.set_xscale("log")
ax.set_yscale("log")
ax.plot(distances / logN, var_mean_envs, label='Environment')
ax.plot(distances / logN, mean_var_envs, label='Sampling')
ax.plot(distances / logN, var_total, label='Variance Total')

xvals = np.array([8, 100])
yvals = 2 * xvals ** 4
yvals2 = 0.75 * xvals ** 3
ax.plot(xvals, yvals, 'k')
ax.plot(xvals, yvals2, 'k')
ax.legend()
ax.grid(True)
ax.set_xlim([0.5, max(distances / logN)])
ax.set_xlabel
fig.savefig("Variance.png", bbox_inches='tight')
