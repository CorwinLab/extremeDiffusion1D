import numpy as np
from pyDiffusion import pyfirstPassageNumba
import time

nExp = 1
N = float(f"1e{nExp}")
dMax = int(1000 * np.log(N))
distances = np.geomspace(1, dMax, 750)
distances = np.unique(distances.astype(int))

time_interval = 3600 * 12

for d in distances:
    quantile = None
    running_sum_squared = 0
    running_sum = 0

    pdf = pyfirstPassageNumba.initializePDF(d)
    firstPassageCDF = pdf[0] + pdf[-1]
    if N==10:
        nFirstPassageCDFPrev = 1 - (1-firstPassageCDF)**N
    else:
        nFirstPassageCDFPrev = 1 - np.exp(-firstPassageCDF * N)
    t = 1
    while (nFirstPassageCDFPrev < 1) or (firstPassageCDF < 1 / N):
        print(nFirstPassageCDFPrev, firstPassageCDF)
        pdf = pyfirstPassageNumba.iteratePDF(pdf)

        firstPassageCDF = pdf[0] + pdf[-1]
        if N==10:
            nFirstPassageCDF = 1 - (1-firstPassageCDF)**N
        else: 
            nFirstPassageCDF = 1 - np.exp(-firstPassageCDF * N)
            
        nFirstPassagePDF = nFirstPassageCDF - nFirstPassageCDFPrev
        
        running_sum_squared += t ** 2 * nFirstPassagePDF
        running_sum += t * nFirstPassagePDF
        if (quantile is None) and (firstPassageCDF > 1 / N):
            quantile = t

        t+=1
        nFirstPassageCDFPrev = nFirstPassageCDF

    variance = running_sum_squared - running_sum ** 2
    print([d, quantile, variance])
    time.sleep(5)