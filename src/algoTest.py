import numpy as np
import npquad 
from libDiffusion import FirstPassagePDF

x = 3
beta = np.inf
staticEnivornment = False 
fpdf = FirstPassagePDF(beta, x, staticEnivornment)

tMax = 10
for _ in range(tMax):
    fpdf.iterateTimeStep()
    print(fpdf.getPDF(), sum(fpdf.getPDF()))

fpdf = FirstPassagePDF(beta, 100, staticEnivornment)
quantile, var, N = fpdf.evolveToCutoffMultiple(1, [10000])
print(N, quantile, var)