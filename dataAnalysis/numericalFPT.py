import numpy as np
from scipy.special import erfc

def cumulativeDistribution(L, t, nTerms=50, verbose=False):
    sum = np.zeros(len(t))
    for k in range(0, nTerms):
        # This is the correct way to do it but we can simplify
        # sum += (-1)**k * erfc(L * np.abs(1 + 2*k)/np.sqrt(2 * t)) * np.sign(1+2*k)
        if verbose:
            print(f'k={k}')
        sum += 2 * (-1) ** k * erfc(L*(1+2*k)/np.sqrt(2*t))
    return sum


def getNParticleMeanVar(positions, N, standardDeviations=10, numpoints=int(1e4), nTerms=50, verbose=False):
    mean = np.zeros(len(positions))
    var = np.zeros(len(positions))
    for i in range(len(positions)):
        logN = np.log(N)
        L = positions[i]
        Nmean = L**2/2/logN
        Nvar = np.pi / 24 *L**4 / logN**4
        tMin = 1
        tMax = Nmean + standardDeviations * np.sqrt(Nvar)
        if tMin < 0:
            tMin = 0
        t = np.linspace(tMin, tMax, numpoints)
        single_particle_cdf = cumulativeDistribution(L, t, nTerms, verbose)
        nParticle_cdf = 1 - np.exp(-N * single_particle_cdf)
        pdf = np.diff(nParticle_cdf)

        mean[i] = np.sum(t[:-1] * pdf)
        var[i] = np.sum(t[:-1]**2 * pdf) - mean[i]**2
    return mean, var