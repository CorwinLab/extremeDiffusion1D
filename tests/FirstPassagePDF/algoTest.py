'''Want to make sure evolveToCutoffMultiple is the same as evolveToCutoff'''

import numpy as np
import npquad 
from pyDiffusion import FirstPassagePDF

x = 3
beta = np.inf
staticEnivornment = False 
fpdf = FirstPassagePDF(beta, x, staticEnivornment)

tMax = 10
for _ in range(tMax):
    fpdf.iterateTimeStep()
    print(fpdf.getPDF(), sum(fpdf.getPDF()), fpdf.getFirstPassageCDF())

N = 10000
fpdf = FirstPassagePDF(beta, 100, staticEnivornment)
quantile, var, Nparticles = fpdf.evolveToCutoffMultiple([N], 1)
print(Nparticles, quantile, var)

fpdf = FirstPassagePDF(beta, 100, staticEnivornment)
quantile, var = fpdf.evolveToCutoff(1, N)
print(N, quantile, var)