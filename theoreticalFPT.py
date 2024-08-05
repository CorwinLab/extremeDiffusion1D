import numpy as np
from matplotlib import pyplot as plt 
from numba import njit
from scipy.special import erfc
import time

@njit
def probabilityDistribution(L, t):
    prefactor = L / np.sqrt(2 * np.pi) / t**(3/2)
    sum = np.zeros(len(t))
    for n in range(-1000, 1000):
        sum += (-1)**(n) * (2 * n + 1) * np.exp(-(L*(2*n+1))**2 / 2 / t)
    return prefactor * sum

def cumulativeDistribution(L, t, nTerms=50, verbose=False):
    sum = np.zeros(len(t))
    for k in range(0, nTerms):
        # This is the correct way to do it but we can simplify
        # sum += (-1)**k * erfc(L * np.abs(1 + 2*k)/np.sqrt(2 * t)) * np.sign(1+2*k)
        # if verbose:
        #     print(f'k={k}')
        sum += 2 * (-1) ** k * erfc(L*(1+2*k)/np.sqrt(2*t))
    return sum

def getNParticlePDF(L, N, standarDeviations=10, numpoints=int(1e4), nTerms=50, verbose=False):
    logN = np.log(N)
    Nmean = L**2/2/logN
    Nvar = np.pi / 24 *L**4 / logN**4
    tMin = 0.001
    tMax = Nmean + standarDeviations * np.sqrt(Nvar)
    if tMin < 0:
        tMin = 0
    t = np.linspace(tMin, tMax, numpoints)
    single_particle_cdf = cumulativeDistribution(L, t, nTerms, verbose)
    nParticle_cdf = 1 - np.exp(-N * single_particle_cdf)
    pdf = np.diff(nParticle_cdf)
    return t[1:], pdf

def gumbelPDF(x, mean, var):
    gamma = 0.577
    beta = -np.sqrt(6 * var / np.pi**2)
    mu = mean - beta*gamma
    z = (x - mu) / beta 
    cdf = 1 - np.exp(-np.exp(-(x-mu)/beta))
    return np.diff(cdf)

def getNParticleMeanVar(positions, N, standarDeviations=10, numpoints=int(1e4), nTerms=50, verbose=False):
    mean = np.zeros(len(positions))
    var = np.zeros(len(positions))
    for i in range(len(positions)):
        logN = np.log(N)
        L = positions[i]
        Nmean = L**2/2/logN
        Nvar = np.pi / 24 *L**4 / logN**4
        tMin = 0.001
        tMax = Nmean + standarDeviations * np.sqrt(Nvar)
        if tMin < 0:
            tMin = 0
        t = np.linspace(tMin, tMax, numpoints)
        single_particle_cdf = cumulativeDistribution(L, t, nTerms, verbose)
        nParticle_cdf = 1 - np.exp(-N * single_particle_cdf)
        pdf = np.diff(nParticle_cdf)

        mean[i] = np.sum(t[:-1] * pdf)
        var[i] = np.sum(t[:-1]**2 * pdf) - mean[i]**2
    return mean, var

if __name__ == '__main__':
    
    N = 1e12
    L = np.geomspace(1, 750 * np.log(N), 500)
    start = time.time()
    mean, var = getNParticleMeanVar(L, N)
    
    fig, ax = plt.subplots()
    #ax.plot(L / np.log(N), mean)
    t0 = L**2 / 2 / np.log(N*np.sqrt(2/np.pi))
    delta = - t0**2 / L**2 * np.log(t0 / L**2)
    ax.plot(L / np.log(N), mean - L**2 /2/np.log(N), c='k')
    ax.plot(L / np.log(N), mean - L**2 / 2 / np.log(N*np.sqrt(2/np.pi)))
    ax.plot(L / np.log(N), np.abs(mean - (t0+delta)), c='tab:orange')
    ax.set_xscale("log")
    ax.set_yscale("log")
    fig.savefig("Mean.png", bbox_inches='tight')
    
    fig, ax = plt.subplots()
    ax.plot(L / np.log(N), var, c='k', label=r'$\mathrm{Var}(\tau_{\mathrm{Sam}})$')
    ax.plot(L / np.log(N), np.pi**2 / 24 * L**4 / np.log(N)**4, c='b', label=r'$\frac{\pi^2 L^4}{24 \log(N)^4}$')
    ax.plot(L / np.log(N), var - np.pi**2 / 24 * L**4 / np.log(N)**4, c='r', label=r'$\mathrm{Var}(\tau_{\mathrm{Sam}}) - \frac{\pi^2 L^4}{24 \log(N)^4}$')
    ax.plot(L / np.log(N), np.abs(var - np.pi**2 / 6 / (L**2 / 2 / (t0+delta)**2)**2), c='tab:green', alpha=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    fig.savefig("Var.png", bbox_inches='tight')
    
    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_yscale("symlog")
    ax.set_xlabel(r"$L / \log(N)$")
    ax.set_ylabel(r'$\mathrm{Var}(\tau_{\mathrm{Sam}}) - \frac{\pi^2 L^4}{24 \log(N)^4}$')
    Ns = [1e5, 1e6, 5e6, 1e7, 1e12, 1e28]
    for N in Ns: 
        L = np.geomspace(np.log(N), 750 * np.log(N), 50)
        mean, var = getNParticleMeanVar(L, N)
        meanFirstTerm, varFirstTerm = getNParticleMeanVar(L, N, nTerms=10, verbose=True)
        ax.plot(L / np.log(N), var - np.pi**2 / 24 * L**4 / np.log(N)**4, label=str(N))
        ax.plot(L / np.log(N), var - varFirstTerm, ls='--', label=str(N)+'First Order')
    ax.set_xlim([1, 750])
    ax.legend()
    fig.savefig("Residual.pdf", bbox_inches='tight')

    Ns = np.geomspace(1e1, 1e250, num=500)
    
    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_yscale("symlog")
    ax.set_xlabel(r"$\log_2(N)$")
    ax.set_ylabel(r"$\mathrm{Var}(\tau_{\mathrm{Sam}}) - \frac{\pi^2 L^4}{24 \log(N)^4}$")
    
    xvals = np.array([10., 100.])
    ax.plot(xvals, 1/xvals**5 * 1e15, ls='--', c='k')
    for L in [500, 2000, 4000, 10000]:
        vars = np.zeros(len(Ns))
        varsFirstOrder = np.zeros(len(Ns))
        for i, N in enumerate(Ns):
            mean, var = getNParticleMeanVar([L], N)
            _, varFirstOrder = getNParticleMeanVar([L], N, nTerms=10, verbose=True)
            vars[i] = var - np.pi**2 / 24 * L**4 / np.log(N)**4
            varsFirstOrder[i] = (var - varFirstOrder)[0]
        ax.plot(np.log2(Ns), vars, label=f'L={L}')
        ax.plot(np.log2(Ns), varsFirstOrder, label=f'L={L} First Order')
        print(f"Finshed {L}")
    ax.legend()
    ax.grid()
    fig.savefig("Residuals.pdf", bbox_inches='tight')

    N=1e12
    L=750*np.log(N)
    t, pdf = getNParticlePDF(L, N)
    mean = np.sum(t * pdf)
    var = np.sum(t**2 * pdf) - mean**2
    print(np.sum(pdf))

    gumbel = gumbelPDF(t, mean, var)
    print(np.sum(gumbel[1:]))
    fig, ax = plt.subplots()
    ax.plot(t, pdf, label='Truth')
    ax.plot(t[1:], gumbel, label='Gumbel Distribution', ls='--')
    ax.vlines(mean, 10**-15, 10**-1, ls='--', label='Mean of Truth')
    ax.set_yscale("log")
    ax.set_ylabel("Probability Density")
    ax.set_xlabel("Time")
    ax.set_xlim([5*10**5, 10**7])
    ax.set_ylim([10**-15, 10**-1])
    ax.legend()
    fig.savefig("PDF.png", bbox_inches='tight')