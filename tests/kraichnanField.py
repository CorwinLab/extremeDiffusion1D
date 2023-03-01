import numpy as np
from matplotlib import pyplot as plt

def getField(positions, fourierCutoff=20):
    field = np.ones(positions.shape)
    k0 = 1
    for i in range(fourierCutoff):
        kn = np.random.normal(0, k0 / np.sqrt(3), size=2)
        v =  np.random.normal(0, 1) * np.array([kn[1], -kn[0]])
        w =  np.random.normal(0, 1) * np.array([kn[1], -kn[0]])
        for pID in range(positions.shape[0]):
            field[pID, :] +=  v * np.cos(np.dot(kn, positions[pID, :])) + w * np.sin(np.dot(kn, positions[pID, :]))
    return field

if __name__ == '__main__':
    # Just get a single field
    x = np.linspace(0, 10, num=50)
    y = np.linspace(0, 10, num=50)
    xx, yy = np.meshgrid(x, y)
    xx = xx.flatten()
    yy = yy.flatten()
    positions = np.vstack([xx, yy]).T
    
    field = getField(positions)
    
    fig, ax = plt.subplots()
    ax.quiver(xx, yy, field[:, 0], field[:, 1], np.sqrt(field[:, 0]**2 + field[:, 1]**2), angles='xy')
    fig.savefig("IncompressibleField.pdf", bbox_inches='tight')
    
    # Calculate two-point correlator
    xmax = 10
    x = np.linspace(0,xmax, num=51)
    y = np.linspace(0,xmax, num=51)
    xx, yy = np.meshgrid(x, y)
    x = xx.flatten()
    y = yy.flatten()
    pos = np.vstack([x, y]).T
    xcenter = xx[xx.shape[0] // 2, xx.shape[1] // 2] 
    ycenter = yy[yy.shape[0] // 2, yy.shape[1] // 2]

    numSystems = 1500
    dot_product =np.zeros(xx.shape)
    for i in range(numSystems): 
        c = getField(pos, fourierCutoff=20)
        c = c.reshape((xx.shape[0], xx.shape[1], 2))
        dot_product += c[xx.shape[0] // 2, xx.shape[1] // 2, 0] * c[:, :, 0] + c[xx.shape[0] // 2, xx.shape[1] // 2, 1] * c[:, :, 1]
        print(i)

    dot_product /= numSystems
    xx = x.reshape(xx.shape)
    yy = y.reshape(yy.shape)

    fig, ax = plt.subplots()
    surf = ax.contourf(xx - xcenter, yy - ycenter, dot_product, cmap=plt.get_cmap('cool'))
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    cbar = fig.colorbar(surf)
    cbar.ax.set_ylabel(r"$\langle \vec{\Psi}(\vec{x}) \vec{\Psi}(\vec{0}) \rangle$")
    fig.savefig("IncompressibleTwoPointCorrelator.pdf")

    def gaussian(x, mean, var):
        return np.exp(-(x - mean)**2 / 2 / var)
    
    xvals = xx[xx.shape[0] // 2, :]

    fig, ax = plt.subplots()
    ax.plot(xvals - xcenter, dot_product[xx.shape[0] // 2, :], c='r')
    ax.set_ylabel(r"$\langle \vec{\Psi}(\vec{x}) \vec{\Psi}(\vec{x}') \rangle$", fontsize=18)
    ax.set_xlabel(r"$|\vec{x}-\vec{x}'|$", fontsize=18)
    fig.savefig("IncompressibleSliceCorrelator.png", bbox_inches='tight')