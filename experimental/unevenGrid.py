import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import lagrange
from pyDiffusion.pydiffusion2D import generateGCF1D

def getSecondDerivative(x, y):
    poly = lagrange(x, y)
    return poly.coef[0]

def getFirstDerivative(x, y):
    poly = lagrange(x, y)
    if len(poly.coef) == 1:
        return 0
    return poly.coef[1]

def iterateTimeStep(x, p, D, dt):
    p_new = np.zeros(p.shape)
    field = generateGCF1D(x, 1, 1)
    prod = field * p
    for i in range(1, len(p_new)-1):
        dp2 = getSecondDerivative(x[i-1:i+1], p[i-1:i+1])
        dp1 = getFirstDerivative(x[i-1:i+1], prod[i-1:i+1])
        p_new[i] = (D * dp2 - dp1) * dt + p[i]
    return p_new
if __name__ == '__main__':
    x = np.linspace(-10**(1/3), 10**(1/3), num=51)
    x = x ** 3
    print(min(np.diff(x)))
    D = 1
    p = np.zeros(x.shape)
    p[p.size // 2] = 1
    fig, ax = plt.subplots()
    ax.scatter(x, p, s=1)
    fig.savefig("Uneven.png")

    p = iterateTimeStep(x, p, D, 0.1)
    fig, ax = plt.subplots()
    ax.scatter(x, p, s=1)
    fig.savefig("Uneven1.png")