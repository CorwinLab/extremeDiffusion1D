import numpy as np 
from matplotlib import pyplot as plt

D = 1
sigma = 0.1
nSystems = 500
v = 1/2
tstar = 2 *(2*D)**9 / sigma**4 / v**8
xstar = (2*D)**5 / sigma**2 / v**4 
y0 = D / v 

'''Set up variables to measure'''
t = tstar * 100
dt = 1
xmeasurement = v * t + y0
num = t / dt # number of random variances to generate

nParticles = 100_000_000

for n in range(nSystems):

    var = np.random.normal(loc = num * D, scale = num * sigma, size=nParticles)
    print(var.min())
    if np.sum(var < 0) != 0:
        raise ValueError

    pos = dt * np.random.normal(loc=0, scale=np.sqrt(var))

    fig, ax = plt.subplots()
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$P(X > x, t)$")
    ax.set_title(fr"$t={t}$")
    ax.set_yscale("log")
    ax.set_xlim([300, max(pos)+5])

    nbins=10000
    lw=0.5

    ax.hist(pos, bins=nbins, cumulative=-1, density=True, histtype='step', lw=lw, label='Random Diffusion Coefficient')

    var = num * D
    pos = dt * np.random.normal(loc=0, scale=np.sqrt(var), size=nParticles)
    ax.hist(pos, bins=nbins, cumulative=-1, density=True, histtype='step', lw=lw, label='Quenched Env')
    ax.legend()
    fig.savefig(f"./Systems/Distribution{n}.pdf", bbox_inches='tight')
    plt.close(fig)