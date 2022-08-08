import numpy as np
import npquad 
from libDiffusion import FirstPassagePDF

x = 3
beta = 1
staticEnivornment = False 
fpdf = FirstPassagePDF(beta, x, staticEnivornment)

tMax = 10
for _ in range(tMax):
    fpdf.iterateTimeStep()
    print(fpdf.getPDF(), sum(fpdf.getPDF()))
