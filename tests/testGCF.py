import numpy as np
from matplotlib import pyplot as plt
from pyDiffusion.pydiffusion2D import getGCF1D

step_size = 0.1

positions = np.arange(-100, 100, step=step_size)
xi = 5
D = np.pi
numSystems = 1000
two_point = None
average_field = None
for j in range(numSystems):
    field = getGCF1D(positions, xi, D, grid_spacing=0.01)
    field_at_middle = field[0]
    if two_point is None:
        two_point = field_at_middle * field[0: ]
    else:
        two_point += field_at_middle * field[0: ]
    
    if average_field is None: 
        average_field = field 
    else:
        average_field += field

average_field = average_field / numSystems
two_point = two_point / numSystems

x = np.arange(0, 200, step=step_size)
theoretical = D * np.exp(-x**2 / 4 / xi**2)
fig, ax = plt.subplots()
ax.plot(x, two_point)
ax.plot(x, theoretical, ls='--', c='k')
ax.set_xlabel(r"$|x-x'|$")
ax.set_ylabel(r"$\langle \xi(x) \xi(x') \rangle$")
ax.set_xlim([0, 200])
fig.savefig("TwoPointCorrelator.pdf", bbox_inches='tight')

fig, ax = plt.subplots()
ax.plot(positions, field)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$\xi(x)$")
fig.savefig("Field.pdf", bbox_inches='tight')

fig, ax = plt.subplots()
ax.plot(positions, average_field)
ax.set_xlabel("Position")
ax.set_ylabel("Average Field Strength")
fig.savefig("AverageField.pdf", bbox_inches='tight')