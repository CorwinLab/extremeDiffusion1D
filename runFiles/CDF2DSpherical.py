import numpy as np
from pyDiffusion.pydiffusionND import iteratePDFSpherical, initiateSphericalPDF

import numpy as np
import npquad 
import csv
import os, psutil
from experimentUtils import saveVars
from datetime import date
import sys 


def runExperiment(N, alpha, Rmin, Rmax, save_file):
    f = open(save_file, 'a')
    writer = csv.writer(f)
    writer.writerow(['Radius', 'Quantile', 'Variance'])
    f.flush()
    
    Rs = np.geomspace(Rmin, Rmax, num=1000).astype(int)
    Rs = np.unique(Rs)

    for R in Rs: 
        pdf = initiateSphericalPDF(R)

        t=0
        quantile = None 
        running_sum_squared = 0 
        running_sum = 0 
        firstPassageCDF = 0 
        nFirstPassageCDFPrev = 1 - np.exp(-N * firstPassageCDF)

        while (nFirstPassageCDFPrev < 1) or (firstPassageCDF < 1/N): 
            pdf, absorbed = iteratePDFSpherical(pdf, R, alpha)
            t += 1

            firstPassageCDF += absorbed
            nFirstPassageCDF = 1 - np.exp(-N * firstPassageCDF)
            nFirstPassagePDF = nFirstPassageCDF - nFirstPassageCDFPrev
            running_sum_squared += t ** 2 * nFirstPassagePDF
            running_sum += t * nFirstPassagePDF

            if (quantile is None) and (firstPassageCDF > 1/N):
                quantile = t 

            nFirstPassageCDFPrev = nFirstPassageCDF
            
        variance = running_sum_squared - running_sum ** 2
        writer.writerow([R, quantile, variance])
    
    f.close()

if __name__ == '__main__':
    # topDir, sysID, alpha, N, Rmin, Rmax = '.', 1, 1, 1e24, 10, 15
    (
        topDir,
        sysID,
        alpha,
        N,
        Rmin,
        Rmax, 
    ) = sys.argv[1:]

    alpha = np.array(4*[float(alpha)])
    N = float(N)
    Rmin = int(Rmin)
    Rmax = int(Rmax)
    save_file=os.path.join(topDir, f"FirstPassage{sysID}.txt")
    
    vars = {"N": N,
            "alpha": alpha,
            "Rmin": Rmin,
            "Rmax": Rmax,
            "save_file": save_file}
    
    vars_file = os.path.join(topDir, "variables.json")
    today = date.today()
    text_date = today.strftime("%b-%d-%Y")

    if int(sysID) == 0:
        vars.update({"Date": text_date})
        saveVars(vars, vars_file)
        vars.pop("Date")
        
    runExperiment(**vars)