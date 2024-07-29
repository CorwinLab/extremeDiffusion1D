import numpy as np
from scipy.special import erf 
import mpmath

def gaussianCumulative(x, mean, var):
	return 1/2 * (1 + erf((x-mean)/np.sqrt(2 * var)))

def J(u, r0, D):
	return u**2 / 4 / D +  2 * r0**2 * u**4 / 3 / (8 * D)**3

def rwreCumulative(u, t, D, r0):
	return 1-np.exp(- t * J(u, r0, D))

def getNParticleMeanVar(times, N, D, model, r0, standardDeviations=10, numpoints=int(1e4)):
	mpmath.mp.dps = 100
	mean = np.zeros(len(times))
	var = np.zeros(len(times))
	for i in range(len(times)):
		logN = np.log(N)
		t = times[i]
		
		Lmean = np.sqrt(4 * D * logN * t)
		Lvar = np.pi**2 / 6 * D * t / logN
		LMin = Lmean - standardDeviations * np.sqrt(Lvar)
		LMax = Lmean + standardDeviations * np.sqrt(Lvar)
		if LMin < 0:
			LMin = 1
		L = np.linspace(LMin, LMax, numpoints)
		
		if model == 'Classical':
			single_particle_cdf = gaussianCumulative(L, 0, 2 * D * t)

		elif model == 'RWRE':
			single_particle_cdf = rwreCumulative(L / t, t, D, r0)

		single_particle_cdf = single_particle_cdf.astype(mpmath.mp.mpf)
		nParticle_cdf = (single_particle_cdf)**mpmath.mp.mpf(N)
		nParticle_cdf = nParticle_cdf.astype(float)
		pdf = np.diff(nParticle_cdf)
		
		mean[i] = np.sum(L[:-1] * pdf)
		var[i] = np.sum(L[:-1]**2 * pdf) - mean[i]**2
		print(i / len(times) * 100)

	return mean, var