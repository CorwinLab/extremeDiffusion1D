import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter


def gaussianPDF(x, mu, sigma):
    return np.exp(-(x - mu)**2 / (2 * sigma**2))


def gaussianSpread(numParticles, width = 1, height = 1, offset = 0):
    y = np.random.rand(numParticles)*height
    x = np.abs(np.random.randn(numParticles)*width)
    plt.plot(x, y + offset, '.')
    xGauss = np.arange(0,20,.01)
    plt.plot(xGauss, 2*gaussianPDF(xGauss, 0, width) + offset + 1.5)
    plt.axis('equal')
    plt.axis('off')
    plt.show()


def quiver(sigma,size=20):
    X = np.arange(size)
    Y = np.arange(size)
    U = gaussian_filter(np.random.randn(len(X), len(Y)), sigma)
    V = gaussian_filter(np.random.randn(len(X), len(Y)), sigma)

    # plt.set_cmap('spring')
    plt.set_cmap('hsv')
    plt.quiver(X, Y, U, V, np.arctan2(U,V))
    plt.axis('equal')
    plt.axis('off')
    plt.show()


def walk2d(NSteps):
    walk = np.cumsum(np.random.randn(NSteps,2),0)
    plt.plot(walk[:,0], walk[:,1])
    return walk

def distWidth(N=1e6, padding = 10):
    time = np.geomspace(1, N)
    extTime = np.geomspace(1/padding, N*padding)
    longTime = np.geomspace(1, N*padding)
    plt.loglog(longTime, longTime**(1/2))
    plt.loglog(time, N**(1/3) * time**(1/6))
    plt.loglog(extTime, N**(1/2)*(1+extTime*0))
    plt.loglog(extTime, N**(1/3)*(1+extTime*0))
    plt.axis('scaled')


def plotWalkers(nWalkers, startPoint, file):
    walkers = np.random.uniform(size=(nWalkers, 2)) * .8 + .1 + startPoint
    plt.plot(walkers[:,0], walkers[:,1], 'k.')
    plt.savefig(file)
    plt.show()

def BCModel(size, nWalkers=100):
    bias = np.random.uniform(size=size)
    midPoint = np.floor(size[1]/2).astype(int)
    plt.pcolormesh(1-bias)
    plt.show()
    plotWalkers(nWalkers, [midPoint, 0], '0.png')

    step1 = np.random.multinomial(nWalkers, [bias[0, midPoint], 1])
    plotWalkers(step1[0], [midPoint, 1], '1.png')
    plotWalkers(step1[1], [midPoint+1, 1], '2.png')

    step2A = np.random.multinomial(step1[0], [bias[1, midPoint], 1])
    step2B = np.random.multinomial(step1[1], [bias[1, midPoint+1], 1])
    plotWalkers(step2A[0], [midPoint, 2], '3.png')
    plotWalkers(step2A[1], [midPoint+1, 2], '4.txt')
    plotWalkers(step2B[0], [midPoint+1, 2], '5.txt')
    plotWalkers(step2B[1], [midPoint+2, 2], '6.txt')

if __name__ == '__main__':
    import matplotlib
    print(matplotlib.get_backend())
    BCModel([500, 500])
