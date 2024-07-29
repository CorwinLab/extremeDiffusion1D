import numpy as np
from matplotlib import pyplot as plt 
from scipy.special import erfc
import mpmath
from tqdm import tqdm

def cumulativeDistribution(L, t, D, nTerms=50, verbose=False):
    sum = np.zeros(len(t))
    for k in range(0, nTerms):
        # This is the correct way to do it but we can simplify
        # sum += (-1)**k * erfc(L * np.abs(1 + 2*k)/np.sqrt(2 * t)) * np.sign(1+2*k)
        # if verbose:
        #     print(f'k={k}')
        sum += 2 * (-1) ** k * erfc(L*(1+2*k)/np.sqrt(4*t*D))
    return sum

def getNParticleMeanVar(positions, N, D, standarDeviations=10, numpoints=int(1e4), nTerms=50, verbose=False):
	mean = np.zeros(len(positions))
	var = np.zeros(len(positions))
	mpmath.mp.dps = 250
	Nmath = mpmath.mp.mpf(N)

	for i in tqdm(range(len(positions))):
		logN = np.log(N)
		L = positions[i]
		Nmean = L**2 / 4 / D / logN
		Nvar = np.pi / 96 / D**2 * L**4 / logN**4
		tMin = 0.001
		tMax = Nmean + standarDeviations * np.sqrt(Nvar)
		if tMin < 0:
			tMin = 0
		t = np.linspace(tMin, tMax, numpoints)
		single_particle_cdf = cumulativeDistribution(L, t, D, nTerms, verbose)
		single_particle_cdf  = single_particle_cdf * mpmath.mp.mpf(1)
		nParticle_cdf = 1 - (1-single_particle_cdf)**N
		pdf = np.diff(nParticle_cdf)
		pdf = pdf.astype(float)

		mean[i] = np.sum(t[:-1] * pdf)
		var[i] = np.sum(t[:-1]**2 * pdf) - mean[i]**2
		
	return mean, var

if __name__ == '__main__': 
	N = 1e2
	L = np.geomspace(1, 750 * np.log(N), 100)
	D = 1/2
	mean, var = getNParticleMeanVar(L, N, D, nTerms=50)

	fig, ax = plt.subplots()
	ax.set_xscale("log")
	ax.set_yscale("log")
	ax.plot(L, var / L**4)
	fig.savefig("Var.png")