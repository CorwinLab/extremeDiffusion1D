import numpy as np
from matplotlib import pyplot as plt
from pyDiffusion.pydiffusion2D import generateGCF2D
plt.rcParams.update({'font.size': 15, 'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts}'})

xi = 3
xmax = 10
sigma = 3
num_points = 2000

x = np.linspace(-xmax, xmax, int(np.sqrt(num_points)))
y = np.linspace(-xmax, xmax, int(np.sqrt(num_points)))
xx, yy = np.meshgrid(x, y)
x = xx.flatten()
y = yy.flatten()
pos = np.vstack([x, y]).T

xcenter = xx[xx.shape[0] // 2, xx.shape[1] // 2] 
ycenter = yy[yy.shape[0] // 2, yy.shape[1] // 2]

# Create a field
c = generateGCF2D(pos, xi, sigma)
c = c.reshape((xx.shape[0], xx.shape[1], 2))

fig, ax = plt.subplots()
ax.set_xlim([-xmax, xmax])
ax.set_ylim([-xmax, xmax])
field = ax.quiver(xx - xcenter, yy - ycenter, c[:, :, 0], c[:, :, 1], np.sqrt(c[:, :, 0]**2 + c[:, :, 1]**2), angles='xy')
cbar = fig.colorbar(field)
cbar.ax.set_ylabel(r"$|\Psi(\vec{x})|$")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_title(fr"$r_c = {xi}, \sigma = {sigma}$")
fig.savefig("Field.pdf", bbox_inches='tight')

numSystems = 500
dot_product =np.zeros(xx.shape)
for i in range(numSystems): 
    c = generateGCF2D(pos, xi, sigma, tol=0.001)
    c = c.reshape((xx.shape[0], xx.shape[1], 2))
    dot_product += c[xx.shape[0] // 2, xx.shape[1] // 2, 0] * c[:, :, 0] + c[xx.shape[0] // 2, xx.shape[1] // 2, 1] * c[:, :, 1]
    print(i)

dot_product = dot_product / numSystems
xx = x.reshape(xx.shape)
yy = y.reshape(yy.shape)

fig, ax = plt.subplots()
surf = ax.contourf(xx - xcenter, yy - ycenter, dot_product, cmap=plt.get_cmap('cool'))
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_title(fr"$r_c = {xi}, \sigma = {sigma}$")
cbar = fig.colorbar(surf)
cbar.ax.set_ylabel(r"$\langle \vec{\Psi}(\vec{x}) \vec{\Psi}(\vec{0}) \rangle$")
fig.savefig("TwoPointCorrelator.pdf")

theoretical = sigma / xi * ( np.exp(-(xcenter - xx[xx.shape[0]//2, :])** 2 / xi**2) / np.pi / xi**2)

fig, ax = plt.subplots()
ax.plot(xx[xx.shape[0]//2, :], dot_product[xx.shape[0] // 2, :])
ax.plot(xx[xx.shape[0]//2, :], theoretical, ls='--', c='k', label=r"$\frac{\sigma}{\xi^2} \frac{e^{-\frac{\vec{x}^2}{\xi^2}}}{\pi \xi^2}$")
ax.set_ylabel(r"$\langle \vec{\Psi}(\vec{x}) \vec{\Psi}(\vec{0}) \rangle$", fontsize=18)
ax.set_xlabel(r"$|\vec{x}|$", fontsize=18)
ax.set_title(fr"$r_c = {xi}, \sigma = {sigma}$")
ax.set_xlim([-xmax, xmax])
ax.legend(fontsize=18)
fig.savefig("SliceCorrelator.pdf", bbox_inches='tight')