import numpy as np
from pyDiffusion import pyfirstPassageNumba, DiffusionTimeCDF
from matplotlib import pyplot as plt
import time
from multiprocessing import Pool, cpu_count

nSystems = 500

def measureFPTCDF(id):
    x = 1000
    pdf = pyfirstPassageNumba.initializePDF(x)

    firstPassageCDF = [0]
    while firstPassageCDF[-1] < 0.99:
        pdf = pyfirstPassageNumba.iteratePDF(pdf)
        firstPassageCDF.append(pdf[0] + pdf[-1])

    print(f"Finished Sys: {id}")
    np.savetxt(f"./FPTCDF/CDF{id}.txt", firstPassageCDF)

def measureFPTDoubleSided(id):
    x = 1000
    tMax = 3_900_000
    cdf = DiffusionTimeCDF('beta', [1, 1], tMax+1)
    firstPassageCDF = np.zeros(shape=tMax)
    for t in range(tMax):
        cdf.iterateTimeStep()
        firstPassageCDF[t] = cdf.getProbOutsidePositions(x)
        print(t / tMax * 100, "%")
    print(f"Finshed Sys: {id}")
    np.savetxt(f"../FPTCDF/DoubleSided/CDF{id}.txt", firstPassageCDF)

ids = np.arange(0, nSystems, dtype=int)

with Pool(cpu_count()-1) as p:
    p.map(measureFPTDoubleSided, ids)
