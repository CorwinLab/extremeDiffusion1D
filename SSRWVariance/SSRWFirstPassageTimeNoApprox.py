import numpy as np
from pyDiffusion import pyfirstPassageNumba
import csv
import os

def runExperiment(Lvalues, N, save_file, model='SSRW'):
    f = open(save_file, "a")
    writer = csv.writer(f)
    writer.writerow(["Position", "Mean", "Variance"])
    f.flush()

    for L in Lvalues: 

        pdf = pyfirstPassageNumba.initializePDF(L)
        firstPassageCDF = pdf[0] + pdf[-1]
        nFirstPassageCDFPrev = 1 - (1-firstPassageCDF)**N
        t = 1
        running_sum_squared = 0
        running_sum = 0
        while (nFirstPassageCDFPrev < 1):
            pdf = pyfirstPassageNumba.iteratePDF(pdf, model='SSRW')

            firstPassageCDF = pdf[0] + pdf[-1]
            nFirstPassageCDF = 1 - (1-firstPassageCDF)**N
            nFirstPassagePDF = nFirstPassageCDF - nFirstPassageCDFPrev
            
            running_sum_squared += t ** 2 * nFirstPassagePDF
            running_sum += t * nFirstPassagePDF

            t+=1
            nFirstPassageCDFPrev = nFirstPassageCDF

        variance = running_sum_squared - running_sum ** 2
        writer.writerow([L, running_sum, variance])
        f.flush()
    f.close()

if __name__ == '__main__':
    Nexps = [2, 5, 12]
    for Nexp in Nexps:
        N = float(f"1e{Nexp}")
        Lvalues = np.geomspace(1, np.log(N)*1000, 750).astype(int)
        Lvalues = np.unique(Lvalues)
        Lvalues = Lvalues[Lvalues <= 750*np.log(N)]
        save_file = os.path.join("./NoApprox", f"MeanVar{Nexp}.txt")
        runExperiment(Lvalues, N, save_file)
