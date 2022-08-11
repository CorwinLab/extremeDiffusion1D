import sys 
from pyDiffusion import FirstPassagePDF, FirstPassageDriver
import time 
import numpy as np
import npquad

distances = np.unique(np.geomspace(60, 1500, 100)).astype(int)
print(distances)
nParticles = 1e24
cutoff = 1
beta = 1

def timeFunc(func, args, samples=10): 
    times = []
    for _ in range(samples):
        start = time.time() 
        func(*args)
        times.append(time.time() - start)
    
    return np.mean(times), np.var(times)

def multipleParticles(nParticles, cutoff, distances, beta):
    for d in distances:
        pdf = FirstPassagePDF(beta, d, False)
        quantiles, variance, Ns = pdf.evolveToCutoffMultiple(nParticles, cutoff)

def singleObject(nParticles, cutoff, distances, beta):
    pdf = FirstPassageDriver(beta, distances)
    pdf.evolveToCutoff(nParticles, 'test.csv', cutoff, writeHeader=False)

mean, var = timeFunc(multipleParticles, [[nParticles], cutoff, distances, beta], samples=5)
print(f"Multiple Particle: {mean} +/- {var}")

mean, var = timeFunc(singleObject, [nParticles, cutoff, distances, beta], samples=5)
print(f"Single Object: {mean} +/- {var}")