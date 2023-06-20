import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import trapz

def generateGCF(pos, correlation_length):

    field = np.zeros(pos.shape[0])

    kx = np.arange(-20, 20, 1)
    ky = np.arange(-20, 20, 1)

    kx_mesh, ky_mesh = np.meshgrid(kx, ky)

    A = np.random.normal(0, 1, size=kx_mesh.shape)
    B = np.random.uniform(0, 2*np.pi, size=kx_mesh.shape)
    for i in range(pos.shape[0]):
        x, y = pos[i, :]
        integrand = np.exp(-(kx_mesh**2 + ky_mesh**2) * correlation_length**2 / 4) * A * np.cos(B + x*kx_mesh + y*ky_mesh)
        integral = trapz([trapz(integrand_x, kx) for integrand_x in integrand], ky)
        field[i] = integral

    return field

def generate2DGCF(pos, correlation_length):
    field_x = generateGCF(pos, correlation_length)
    field_y = generateGCF(pos, correlation_length)
    return np.array([field_x, field_y]).T

if __name__ == '__main__': 
    xmax = 10
    xi = 2

    x = np.arange(0,xmax,.1)
    y = np.arange(0,xmax,.1)
    xx, yy = np.meshgrid(x, y)
    x = xx.flatten()
    y = yy.flatten()
    pos = np.vstack([x, y]).T

    numSystems = 50
    dot_product =np.zeros(x.shape)
    for i in range(numSystems): 
        c = generate2DGCF(pos, xi)
        dot_product += c[0, 0] * c[:, 0] + c[0, 1] * c[:, 1]
        print(i)

    dot_product /= numSystems
    dot_product = dot_product.reshape(xx.shape)

    theoretical = np.max(dot_product)*np.exp(-(x[0] - xx)** 2/ 2 / xi**2)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(xx, yy, dot_product, linewidth=0, antialiased=True, cmap=plt.get_cmap('cool'), alpha=0.75)
    ax.plot3D(xx[0, :], yy[0, :], dot_product[0, :], c='r')
    fig.savefig("TwoPointCorrelator.pdf", bbox_inches='tight')

    theoretical = np.max(dot_product[0, :]) * np.exp(-(xx[0, 0] - xx[0, :])** 2 / (xi)**2)

    fig, ax = plt.subplots()
    ax.plot(xx[0, :], (dot_product[0, :] + dot_product[:, 0]) / 2, c='r')
    ax.plot(xx[0, :], theoretical, ls='--', c='k')
    fig.savefig("SliceCorrelator.png", bbox_inches='tight')
    