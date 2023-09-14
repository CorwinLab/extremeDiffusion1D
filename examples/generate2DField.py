from matplotlib import pyplot as plt
import numpy as np
from pyDiffusion.pydiffusion2D import generateGCF2D
import time

plt.rcParams.update({"font.size": 15, "text.usetex": True, "text.latex.preamble": r"\usepackage{amsfonts}"})

xi = 2
xmax = 10
num_points = int(1e4)
x = np.linspace(-xmax, xmax, int(np.sqrt(num_points)))
y = np.linspace(-xmax, xmax, int(np.sqrt(num_points)))

xx, yy = np.meshgrid(x, y)
x = xx.flatten()
y = yy.flatten()
print(len(x))
pos = np.vstack([x, y]).T

c = generateGCF2D(pos, xi=xi, sigma=5)
start = time.time()
c = generateGCF2D(pos, xi, 5)
print(time.time() - start)
c = c.astype("float64")

fig, ax = plt.subplots()
ax.set_title(fr"$\xi = {xi}$")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_xlim([min(x), max(x)])
ax.set_ylim([min(y), max(y)])
qp = ax.quiver(pos[:, 0], pos[:, 1], c[:, 0], c[:, 1], np.sqrt(c[:, 0]**2 + c[:, 1]**2), angles="xy")

cbar = fig.colorbar(qp, ax=ax)
cbar.ax.set_ylabel(r"$| \vec{\psi}(\vec{x}) |$")
fig.savefig(f"GCF{xi}.pdf", bbox_inches="tight")