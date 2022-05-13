import numpy as np
import glob 
import os 
from matplotlib import pyplot as plt

directories = "/home/jacob/Desktop/talapasMount/JacobData/ParallelFirstPassE/24/"
files = glob.glob(os.path.join(directories,'F*.txt'))

data = np.loadtxt(files[0])

squared = np.zeros(data.shape[1])
added = np.zeros(data.shape[1])

for f in files: 
    data = np.loadtxt(f)
    added += np.sum(data, axis=0)
    squared += np.sum(data**2, axis=0)

num_of_systems = len(files) * data.shape[0]
mean = added / num_of_systems
var = squared / num_of_systems - mean ** 2

numParticles = float(f'1e{24}')
logN = np.log(numParticles)
log2N = np.log2(numParticles)

max_distance = 2000
num_of_save_distances = 1500
distances = np.geomspace(1, max_distance, num_of_save_distances, dtype=np.int64)
distances = np.unique(distances)

fig, ax = plt.subplots()
ax.set_xlabel(r"$x / log_2(N)$")
ax.set_ylabel(r"$Mean(\tau) (\tau=\mathrm{Passage Time})$")
ax.set_title("N=1e24")
ax.plot(distances / log2N, mean)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim([1 / log2N, max(distances)/log2N])
ax.set_ylim([1, 5*10**4])
ax.grid(True)
fig.savefig("Mean.png", bbox_inches='tight')

fig, ax = plt.subplots()
ax.set_xlabel(r"$x / log_2(N)$")
ax.set_ylabel(r"$Var(\tau) (\tau=\mathrm{Passage Time})$")
ax.set_title("N=1e24")
ax.plot(distances / log2N, var)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim([0.5, max(distances)/log2N])
ax.grid(True)
fig.savefig("Var.png", bbox_inches='tight')

directories = "/home/jacob/Desktop/talapasMount/JacobData/ParallelFirstPassE/"
folders = os.listdir(directories)
Ns = []
means = []
vars = []
for f in folders: 
    files = glob.glob(os.path.join(directories, f, 'F*.txt'))
    Ns.append(float(f"1e{int(f)}"))
    data = np.loadtxt(files[0])

    squared = np.zeros(data.shape[1])
    added = np.zeros(data.shape[1])

    for f in files: 
        data = np.loadtxt(f)
        added += np.sum(data, axis=0)
        squared += np.sum(data**2, axis=0)

    num_of_systems = len(files) * data.shape[0]
    mean = added / num_of_systems
    var = squared / num_of_systems - mean ** 2
    means.append(mean)
    vars.append(var)


fig, ax = plt.subplots()
ax.set_xlabel(r"$x / log_2(N)$")
ax.set_ylabel(r"$Var(\tau)$")
ax.set_xscale("log")
ax.set_yscale("log")
for i in range(len(Ns)): 
    N = Ns[i]
    log2N = np.log2(N)
    var = vars[i]
    ax.plot(distances / log2N, var, label=N)
ax.legend()
ax.set_xlim([0.5, max(distances)/np.log2(100)])
ax.grid(True)
fig.savefig("VarMultiple.png")

fig, ax = plt.subplots()
ax.set_xlabel(r"$x / log_2(N)$")
ax.set_ylabel(r"$Mean(\tau)$")
ax.set_xscale("log")
ax.set_yscale("log")
for i in range(len(Ns)): 
    N = Ns[i]
    log2N = np.log2(N)
    mean = means[i]
    ax.plot(distances / log2N, mean, label=N)
ax.legend()
ax.grid(True)
fig.savefig("MeanMultiple.png")