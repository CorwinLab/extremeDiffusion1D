import numpy as np 
from pyDiffusion.pycontinuous1D import generateGCF1D
from matplotlib import pyplot as plt 
plt.rcParams.update({'font.size': 15, 'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts}'})

L = 50
xi = 5
sigma = 5
step = 0.1
x = np.arange(-L, L+step, step=step)

field = generateGCF1D(x, xi, sigma, tol=0.001)
fig, ax = plt.subplots()
ax.plot(x, field)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$\Psi(x)$")
ax.set_title(fr"$\sigma = {sigma}, r_c = {xi}$")
ax.set_xlim([min(x), max(x)])
fig.savefig("1DField.pdf", bbox_inches='tight')

numFields = 5000
dot_product = np.zeros(x.shape)

for i in range(numFields):
    field = generateGCF1D(x, xi, sigma, tol=1e-4)
    dot_product += field * field[len(field) // 2]
    print(i)

dot_product /= numFields
theoretical = np.exp(-x**2 / xi**2)  / np.sqrt(np.pi * xi**2) * sigma / xi

fig, ax = plt.subplots()
ax.plot(x, dot_product)
ax.plot(x, theoretical, ls='--', c='k', label=r'$\frac{\sigma}{\xi} \frac{e^{-\frac{x^2}{r_c^2}}}{\sqrt{\pi r_c^2}}$')
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$\langle\Psi(x, t)\Psi(0, t)\rangle$")
ax.set_title(fr"$\sigma = {sigma}, r_c = {xi}$")
ax.set_xlim([min(x), max(x)])
ax.legend()
fig.savefig("1DDotProduct.pdf", bbox_inches='tight')