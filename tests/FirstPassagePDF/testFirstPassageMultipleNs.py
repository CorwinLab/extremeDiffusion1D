from regex import F
from libDiffusion import FirstPassagePDF
from pyDiffusion.quadMath import prettifyQuad
import numpy as np
import npquad

nParticles = [1e24, 1e50, 1e300]
cutoff = 1

print("Single Particle")
for N in nParticles:
    pdf = FirstPassagePDF(np.inf, 500, False)
    quantileTime, var = pdf.evolveToCutoff(cutoff, N)
    print(N, quantileTime, var)

print("\nMultiple Particle")

pdf = FirstPassagePDF(np.inf, 500, False)
quantiles, variance, Ns = pdf.evolveToCutoffMultiple(cutoff, nParticles)

for q, v, N in zip(quantiles, variance, Ns):
    print(N, q, v)