import numpy as np 
from pyDiffusion.pydiffusion2D import generateGCF1D
from matplotlib import pyplot as plt 

L = 100
xi = 4
sigma = 2
step = 0.1
x = np.arange(-L, L+step, step=step)

field = generateGCF1D(x, xi, sigma, tol=0.001)
fig, ax = plt.subplots()
ax.plot(x, field)
fig.savefig("1DField.png", bbox_inches='tight')

numFields = 5000
dot_product = np.zeros(x.shape)

for i in range(numFields):
    field = generateGCF1D(x, xi, sigma, tol=0.000000001)
    dot_product += field * field[len(field) // 2]
    print(i)

dot_product /= numFields
theoretical = np.exp(-x**2 / xi**2)  / np.sqrt(np.pi * xi**2) * sigma / xi

fig, ax = plt.subplots()
ax.plot(x, dot_product)
ax.plot(x, theoretical, ls='--')
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$\langle\xi(x, t)\xi(0, t)\rangle$")
ax.set_xlim([-L, L])
fig.savefig("1DDotProduct.pdf", bbox_inches='tight')