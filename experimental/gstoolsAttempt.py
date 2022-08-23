from gstools import Gaussian, SRF
import numpy as np
from matplotlib import pyplot as plt

sigma = 1
scale = 1
dim = 2
model = Gaussian(2, 1, 10)
srf = SRF(model, generator='VectorField')
x = y = range(100)
field = srf((x, y), mesh_type='structured')
fig, ax = plt.subplots()
srf.plot(fig=fig)
fig.savefig("Test.pdf")

sigma = 1
scale = 1
dim = 2
model = Gaussian(2, 1, 1)
srf = SRF(model, generator='VectorField')
particle_coords = np.random.uniform(-10, 10, size=(500, 2))
x = particle_coords[:, 0]
y = particle_coords[:, 1]
field = srf((x, y), mesh_type='unstructured')
fig, ax = plt.subplots()
ax.scatter(x, y, s=2)
ax.quiver(x, y, field[0, :], field[1, :])
fig.savefig("GSField.pdf", bbox_inches='tight')