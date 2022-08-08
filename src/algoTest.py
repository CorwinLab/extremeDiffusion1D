import numpy as np
import npquad 
from libDiffusion import iteratePDF

x = 3 
pdf = np.zeros(shape=x+2)
pdf[0] = 1
t = 0

for _ in range(10):
    pdf = iteratePDF(pdf, x, t)
    t+=1 
    print(pdf, sum(pdf))