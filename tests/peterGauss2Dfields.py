import numpy as np
from matplotlib import pyplot as plt
from pyDiffusion.pydiffusion2D import generateGCF

xmax = 10
xi =3
x = np.arange(0,xmax,.1)
y = np.arange(0,xmax,.1)
xx, yy = np.meshgrid(x, y)
x = xx.flatten()
y = yy.flatten()
pos = np.vstack([x, y]).T
c = generateGCF(pos, xi=xi, tol=0.001)
c = c.astype('float64')
fig, ax = plt.subplots()
qp = ax.quiver(pos[:, 0], pos[:, 1], c[:, 0], c[:, 1], np.sqrt(c[:, 0]**2 + c[:, 1]**2), angles='xy')
ax.set_title(f"xi = {xi}")
fig.colorbar(qp, ax=ax)
fig.savefig(f"GCF{xi}.pdf", bbox_inches="tight")

factor=1
xi = 3 * factor 
xmax = 10 * factor 
step = 0.25 * factor

x = np.arange(-xmax, xmax, step=step)
y = np.arange(-xmax, xmax, step=step)
xx, yy = np.meshgrid(x, y)
x = xx.flatten()
y = yy.flatten()
pos = np.vstack([x, y]).T

xcenter = xx[xx.shape[0] // 2, xx.shape[1] // 2] 
ycenter = yy[yy.shape[0] // 2, yy.shape[1] // 2]

numSystems = 1000
dot_product =np.zeros(xx.shape)
for i in range(numSystems): 
    c = generateGCF(pos, xi=xi, tol=0.00001)
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
cbar = fig.colorbar(surf)
cbar.ax.set_ylabel(r"$\langle \vec{\Psi}(\vec{x}) \vec{\Psi}(\vec{0}) \rangle$")
fig.savefig("TwoPointCorrelator.pdf")

theoretical = 1 / xi**2 * np.exp(-(xcenter - xx[xx.shape[0]//2, :])** 2 / xi**2)

fig, ax = plt.subplots()
ax.plot(xx[xx.shape[0]//2, :], dot_product[xx.shape[0] // 2, :], c='r')
ax.plot(xx[xx.shape[0]//2, :], theoretical, ls='--', c='b', label=r"$\frac{1}{\xi^2} e^{-\frac{(\vec{x}-\vec{x}')^2}{\xi^2}}$")
ax.set_ylabel(r"$\langle \vec{\Psi}(\vec{x}) \vec{\Psi}(\vec{x}') \rangle$", fontsize=18)
ax.set_xlabel(r"$|\vec{x}-\vec{x}'|$", fontsize=18)
ax.set_title(fr"$\xi={xi}$", fontsize=18)
ax.set_xlim([-xmax, xmax])
ax.legend(fontsize=18)
fig.savefig("SliceCorrelator.png", bbox_inches='tight')