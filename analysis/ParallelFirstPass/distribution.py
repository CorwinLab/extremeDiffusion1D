import numpy as np
import glob
import matplotlib.pyplot as plt

files = glob.glob('/home/jacob/Desktop/talapasMount/JacobData/ParallelFirstPass/24/F*.txt')

data = np.loadtxt(files[0])
final_positions = data[:, -10]
final_positions = final_positions - np.mean(final_positions)
total_data = final_positions

for i, f in enumerate(files[0:]): 
    data = np.loadtxt(f)
    final_positions = data[:, -10]
    final_positions = final_positions - np.mean(final_positions)
    total_data = np.append(total_data, final_positions)

def gumbel(x, mu, beta):
     z = (-x - mu) / beta 
     return 1/beta * np.exp(-(z+np.exp(-z)))

xvals = np.linspace(min(total_data), max(total_data))
var = np.var(total_data)
mean = np.mean(total_data)
print(mean)
beta = np.sqrt(6 / np.pi**2 * var)
mu = mean - beta * 0.577

gumbel_dist = gumbel(xvals, mu, beta)

fig, ax = plt.subplots()
ax.set_yscale("log")
ax.set_ylabel("Probability Density")
ax.set_xlabel(r"$\tau - \langle \tau \rangle_{\mathrm{Env}}$")
ax.hist(total_data, density=True, bins=50)
ax.plot(xvals, gumbel_dist)
ax.set_ylim([10**-8, 10**-3])
fig.savefig("OverallDistribution.png")