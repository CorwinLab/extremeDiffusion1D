import numpy as np 
import npquad 
import time 
from pyDiffusion import FirstPassageDriver, FirstPassageEvolve 

nParticles = np.quad("1e24")
beta=1
distances = [500, 750, 1000, 1500, 2000]

def timeFunc(func, args, samples=10): 
    times = []
    for _ in range(samples):
        start = time.time() 
        func(*args)
        times.append(time.time() - start)
    
    return np.mean(times), np.var(times)

def evolve(beta, maxPositions, nParticles):
    pdf = FirstPassageEvolve(beta, maxPositions, nParticles)
    pdf.evolveToCutoff('TestEvolve.txt')
    print("Sample Done")

def driver(beta, maxPositions, nParticles):
    pdf = FirstPassageDriver(beta, maxPositions)
    pdf.evolveToCutoff(nParticles, 'TestDriver.txt')
    print("Sample Driver Done")
mean, var = timeFunc(evolve, [beta, distances, nParticles])
print(f"Evolve (Python): {mean} +/- {var}")

mean, var = timeFunc(driver, [beta, distances, nParticles])
print(f"Driver (C++): {mean} +/- {var}")