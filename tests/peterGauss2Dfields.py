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
c = generateGCF(pos, xi=xi, fourierCutoff=20)
c = c.astype('float64')
fig, ax = plt.subplots()
qp = ax.quiver(pos[:, 0], pos[:, 1], c[:, 0], c[:, 1], np.sqrt(c[:, 0]**2 + c[:, 1]**2), angles='xy')
ax.set_title(f"xi = {xi}")
fig.colorbar(qp, ax=ax)
fig.savefig(f"GCF{xi}.pdf", bbox_inches="tight")


xi = 3
xmax = 10

x = np.linspace(0,xmax, num=50)
y = np.linspace(0,xmax, num=50)
xx, yy = np.meshgrid(x, y)
x = xx.flatten()
y = yy.flatten()
pos = np.vstack([x, y]).T

numSystems = 500
dot_product =np.zeros(x.shape)
for i in range(numSystems): 
    c = generateGCF(pos, xi=xi, fourierCutoff=20)
    dot_product += c[0, 0] * c[:, 0] + c[0, 1] * c[:, 1]
    print(i)

dot_product /= numSystems
dot_product = dot_product.reshape(xx.shape)
xx = x.reshape(xx.shape)
yy = y.reshape(yy.shape)

fig, ax = plt.subplots()
surf = ax.contourf(xx, yy, dot_product, cmap=plt.get_cmap('cool'))
ax.set_xlabel(r"$|x-x'|$")
ax.set_ylabel(r"$|y-y'|$")
cbar = fig.colorbar(surf)
cbar.ax.set_ylabel(r"$\langle \vec{\Psi}(\vec{x}) \vec{\Psi}(\vec{x}') \rangle$")
fig.savefig("TwoPointCorrelator.pdf")

theoretical2 = 1 / xi**2 * np.exp(-(xx[0, 0] - xx[0, :])** 2 / xi**2)

fig, ax = plt.subplots()
ax.plot(xx[0, :], dot_product[0, :], c='r')
#ax.plot(xx[0, :], theoretical, ls='--', c='k')
ax.plot(xx[0, :], theoretical2, ls='--', c='b', label=r"$\frac{1}{\xi^2} e^{-\frac{(\vec{x}-\vec{x}')^2}{\xi^2}}$")
ax.set_ylabel(r"$\langle \vec{\Psi}(\vec{x}) \vec{\Psi}(\vec{x}') \rangle$", fontsize=18)
ax.set_xlabel(r"$|\vec{x}-\vec{x}'|$", fontsize=18)
ax.set_title(fr"$\xi={xi}$", fontsize=18)
ax.set_xlim([0, xmax])
ax.legend(fontsize=18)
fig.savefig("SliceCorrelator.png", bbox_inches='tight')