import sys 
sys.path.append("../../src")
from libDiffusion import FirstPassagePDF, FirstPassageDriver
import time 
import numpy as np
import npquad

distances = np.unique(np.geomspace(60, 5000).astype(int))
t = 10000
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

def multipleParticles(nParticles, cutoff, distances, beta, t):
    for d in distances:
        pdf = FirstPassagePDF(beta, d, False)
        while pdf.getTime() < t: 
            pdf.iterateTimeStep()

def singleObject(nParticles, cutoff, distances, beta, t):
    pdf = FirstPassageDriver(beta, distances)
    while pdf.getTime() < t:
        pdf.iterateTimeStep()

mean, var = timeFunc(multipleParticles, [[nParticles], cutoff, distances, beta, t], samples=1)
print(f"Multiple Particle: {mean} +/- {var}")

mean, var = timeFunc(singleObject, [nParticles, cutoff, distances, beta, t], samples=1)
print(f"Single Object: {mean} +/- {var}")