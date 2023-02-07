import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
import json
from pyDiffusion.pydiffusion2D import getGCF1D

file = '/home/jacob/Desktop/talapasMount/JacobData/ContinuousNewLonger/0.1/ParticlePositions991.txt'
positions = np.loadtxt(file)
df = pd.read_csv('/home/jacob/Desktop/talapasMount/JacobData/ContinuousNewLonger/0.1/MaxPositions991.txt')

vars_file = '/home/jacob/Desktop/talapasMount/JacobData/ContinuousNewLonger/0.1/variables.json'
with open(vars_file) as f:
    vars = json.load(f)

N = vars['nParticles']
D = vars['D']
rc = vars['xi']
sigma = vars['sigma']

Dr = D + sigma / rc**2 / np.sqrt(2*np.pi)
r0 = sigma / Dr
t = df['Time'].values[-1]
def gaussian(x, mean, var):
    return 1 / np.sqrt(2 * np.pi * var) * np.exp(-(x-mean)**2 / 2 / var)

# Plot of particle positions
xvals = np.linspace(min(positions), max(positions), 500)
theoretical = gaussian(xvals, 0, 2 * Dr * t)
print("PDF Sum:", sum(theoretical * np.diff(xvals)[0]))
fig, ax = plt.subplots()
ax.hist(positions, bins=250, density=True)
ax.plot(xvals, theoretical, ls='--', c='k')
ax.set_ylim([10**-8, 10**-2])
ax.set_yscale("log")
ax.set_xlabel(r"$x$")
ax.set_ylabel("Number of Particles")
ax.set_title(f"D = {D}, rc = {rc}, sigma={sigma}")
fig.savefig("ParticlePositions.pdf", bbox_inches='tight')

# Plot making a field
field = getGCF1D(positions, rc, sigma)
df = pd.DataFrame(np.array([field, positions]).T, columns=['Field', 'Positions'])
df.sort_values(by='Positions', inplace=True, ignore_index=True)

fig, ax = plt.subplots()
ax.scatter(df['Positions'], df['Field'], s=0.5)
fig.savefig("Field.png", bbox_inches='tight')

# Plot of correlation function
numSystems = 500
grid_spacing=0.1
two_point = None
average_field = None
abs_field = None
for j in range(numSystems):
    field = getGCF1D(positions, rc, sigma, grid_spacing=grid_spacing)
    
    df = pd.DataFrame(np.array([field, positions]).T, columns=['Field', 'Positions'])
    df.sort_values(by='Positions', inplace=True, ignore_index=True)
    df = df[df['Positions'] >= 0]
    field = df['Field'].values
    sorted_positions = df['Positions'].values
 
    field_at_middle = field[0]
    
    if two_point is None:
        two_point = field_at_middle * field
    else:
        two_point += field_at_middle * field
    
    if average_field is None: 
        average_field = field 
    else:
        average_field += field

    if abs_field is None:
        abs_field = np.abs(field)
    else: 
        abs_field += np.abs(field)
    print(j)

abs_field = abs_field / numSystems
average_field = average_field / numSystems
two_point = two_point / numSystems
diff_positions = abs(sorted_positions - sorted_positions[0])

df = pd.DataFrame(np.array([diff_positions, two_point]).T, columns=['Positions', 'Correlator'])
df.sort_values(by='Positions', inplace=True, ignore_index=True)

maxDiff = 1000
x = np.linspace(0, maxDiff, 500)
theoretical = sigma / rc / np.sqrt(2 * np.pi * rc**2) * np.exp(-x**2 / 2 / rc**2)
fig, ax = plt.subplots()
ax.plot(df['Positions'], df['Correlator'], alpha=0.75)
ax.plot(x, theoretical, ls='--', c='k')
ax.set_xlabel(r"$|x-x'|$")
ax.set_ylabel(r"$\langle \xi(x) \xi(x') \rangle$")
ax.set_title(fr"$r_c={rc}, D={D}, dx={grid_spacing}$")
ax.set_xlim([0, maxDiff])
ax.legend()
fig.savefig(f"TwoPointCorrelatorLargePosition{grid_spacing}.png", bbox_inches='tight')

maxDiff = 50
x = np.linspace(0, maxDiff, 500)
theoretical = sigma / rc / np.sqrt(2 * np.pi * rc**2) * np.exp(-x**2 / 2 / rc**2)
fig, ax = plt.subplots()
ax.plot(df['Positions'], df['Correlator'], alpha=0.75)
ax.plot(x, theoretical, ls='--', c='k')
ax.set_xlabel(r"$|x-x'|$")
ax.set_ylabel(r"$\langle \xi(x) \xi(x') \rangle$")
ax.set_title(fr"$r_c={rc}, D={D}, dx={grid_spacing}$")
ax.set_xlim([0, maxDiff])
ax.legend()
fig.savefig(f"TwoPointCorrelator{grid_spacing}.png", bbox_inches='tight')

fig, ax = plt.subplots()
ax.plot(sorted_positions, average_field / abs_field)
ax.hlines(np.mean(average_field / abs_field), 0, max(sorted_positions), ls='--', color='r')
ax.set_xlim([0, max(sorted_positions)])
ax.set_xlabel("x")
ax.set_ylabel(r"$\frac{\langle \xi(x) \rangle}{\langle |\xi(x)| \rangle}$")
fig.savefig("AverageField.png", bbox_inches='tight')