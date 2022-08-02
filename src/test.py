from libDiffusion import FirstPassagePDFMain
import numpy as np 
import npquad

driver = FirstPassagePDFMain(1, [2, 10])
for _ in range(100): 
    driver.iterateTimeStep()
    for d in driver.getPDFs():
        print(sum(d.getPDF()))