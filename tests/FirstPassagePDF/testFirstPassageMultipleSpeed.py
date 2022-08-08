from regex import F
from pyDiffusion import FirstPassagePDF
from pyDiffusion.quadMath import prettifyQuad
import numpy as np
import npquad
import time

nParticles = [1e24, 1e50, 1e300]
cutoff = 1
distances = [500, 600, 750, 1000]

def timeFunc(func, args, samples=10): 
    times = []
    for _ in range(samples):
        start = time.time() 
        func(*args)
        times.append(time.time() - start)
    
    return np.mean(times), np.var(times)

def singleParticle(nParticles, cutoff, distances):
    for d in distances:
        for N in nParticles:
            pdf = FirstPassagePDF(np.inf, d, False)
            quantileTime, var = pdf.evolveToCutoff(cutoff, N)

def multipleParticles(nParticles, cutoff, distances):
    for d in distances:
        pdf = FirstPassagePDF(np.inf, d, False)
        quantiles, variance, Ns = pdf.evolveToCutoffMultiple(nParticles, cutoff)

mean, var = timeFunc(singleParticle, [nParticles, cutoff, distances])
print(f"Single Particle: {mean} +/- {var}")

mean, var = timeFunc(multipleParticles, [nParticles, cutoff, distances])
print(f"Multiple Particle: {mean} +/- {var}")