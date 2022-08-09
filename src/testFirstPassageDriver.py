import numpy as np
import npquad 
from libDiffusion import FirstPassageDriver

maxPositions = [3, 4, 5]
beta = 1

pdf = FirstPassageDriver(beta, maxPositions)
max_time = 5
for i in range(max_time):
    print(f"i:")
    pdf.iterateTimeStep()
    pdfs = pdf.getPDFs()
    for p in pdfs: 
        print(p.getPDF(), sum(p.getPDF()))
    print("\n")