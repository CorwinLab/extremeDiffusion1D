import numpy as np
import npquad 
from libDiffusion import FirstPassagePDF

beta = 1
maxPosition = 1000
staticEnvironment=False 
N = 1e24

fpdf = FirstPassagePDF(beta, maxPosition, staticEnvironment)
fpdf.setBetaSeed(0)

quantile, var, N = fpdf.evolveToCutoffMultiple(1, [N])
print(quantile, var, N)