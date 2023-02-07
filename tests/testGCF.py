import numpy as np
from matplotlib import pyplot as plt
from pyDiffusion.pydiffusion2D import getGCF1D

step_size = 0.1

positions = np.arange(-100, 100, step=step_size)
xi = 5
sigma = 1
numSystems = 1000
grid_spacing = 0.05
two_point = None
average_field = None
abs_field = None
for j in range(numSystems):
    field = getGCF1D(positions, xi, sigma, grid_spacing=grid_spacing)
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

abs_field = abs_field / numSystems
average_field = average_field / numSystems
two_point = two_point / numSystems

x = np.arange(0, 200, step=step_size)
theoretical = sigma / xi / np.sqrt(2 * np.pi * xi**2) * np.exp(-x**2 / 2 / xi**2)
fig, ax = plt.subplots()
ax.plot(x, two_point)
ax.plot(x, theoretical, ls='--', c='k')
ax.set_xlabel(r"$|x-x'|$")
ax.set_ylabel(r"$\langle \xi(x) \xi(x') \rangle$")
ax.set_title(fr"$r_c={xi}, \sigma={sigma}$")
ax.set_xlim([0, 200])
ax.legend()
fig.savefig("TwoPointCorrelator.pdf", bbox_inches='tight')

fig, ax = plt.subplots()
ax.plot(positions, field)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$\xi(x)$")
fig.savefig("Field.pdf", bbox_inches='tight')

fig, ax = plt.subplots()
ax.plot(positions, average_field / abs_field)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$\frac{\langle \xi(x) \rangle}{\langle |\xi(x)| \rangle}$")
ax.set_title(fr"$r_c={xi}, \sigma={sigma}$")
fig.savefig("AverageField.pdf", bbox_inches='tight')