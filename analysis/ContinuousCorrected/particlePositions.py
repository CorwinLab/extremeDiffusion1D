import numpy as np
from matplotlib import pyplot as plt 
import glob

'''
files = glob.glob("/home/jacob/Desktop/corwinLabMount/CleanData/Continuous/1/20/ParticlePositions*.txt")

particle_positions = np.array([])
for i, f in enumerate(files):
    positions = np.loadtxt(f)
    particle_positions = np.hstack([positions, particle_positions])
    print(i)

np.savetxt("ParticlePositions-1-20.txt", particle_positions)
'''
def gaussuianPDF(x, mean, var):
    return 1 / np.sqrt(2 * np.pi * var) * np.exp(-1/2 * (x- mean)**2 / var)

D = 20
t = 10000+1
mean = 0
var = 4 * D * t

positions = np.loadtxt("ParticlePositions-1-20.txt", max_rows=10000000)
x = np.linspace(min(positions), max(positions), 1000)
pdf = gaussuianPDF(x, mean, var)
print(sum(pdf * np.diff(x)[0]))

fig, ax = plt.subplots()
ax.set_yscale("log")
ax.hist(positions, bins=100, density=True)
ax.plot(x, pdf, ls='--', c='k')
fig.savefig("ParticlePositions.pdf")