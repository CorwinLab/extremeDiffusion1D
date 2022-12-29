import numpy as np
from pyDiffusion.pydiffusion2D import generateGCF1D
from matplotlib import pyplot as plt

pos = np.arange(-10, 10, 0.1)
xi = 1
field = generateGCF1D(pos, xi)

fig, ax = plt.subplots()
ax.scatter(pos, field)
ax.set_xlabel("Position")
ax.set_ylabel("Bias")
fig.savefig("Field.png", bbox_inches='tight')