import numpy as np
from pyDiffusion import pyfirstPassageNumba
import os
import csv
import sys
from datetime import date
import time
from experimentUtils import saveVars


def runExperiment(nMin, nMax, distance, num_of_points, save_dir, sysID):
    Nexps = np.unique(np.geomspace(nMin, nMax, num_of_points).astype(int))
    Ns = np.array([float(f"1e{nExp}") for nExp in Nexps])
    write_header = True
    save_file = os.path.join(save_dir, f"FirstPassageCDF{sysID}.txt")
    distances = distance * np.log(Ns)

    f = open(save_file, "a")
    writer = csv.writer(f)
    if write_header:
        writer.writerow(["N", "Quantile", "Variance", "Distance"])
    
    time_interval = 3600 * 12

    for d, N in zip(distances, Ns):
        quantile = None
        running_sum_squared = 0
        running_sum = 0

        pdf = pyfirstPassageNumba.initializePDF(d)
        firstPassageCDF = pdf[0] + pdf[-1]
        nFirstPassageCDFPrev = 1 - np.exp(-firstPassageCDF * N)
        t = 1
        last_save_time = time.time()
        while (nFirstPassageCDFPrev < 1) or (firstPassageCDF < 1 / N):
            pdf = pyfirstPassageNumba.iteratePDF(pdf)

            firstPassageCDF = pdf[0] + pdf[-1]
            nFirstPassageCDF = 1 - np.exp(-firstPassageCDF * N)
            nFirstPassagePDF = nFirstPassageCDF - nFirstPassageCDFPrev
            
            running_sum_squared += t ** 2 * nFirstPassagePDF
            running_sum += t * nFirstPassagePDF
            if (quantile is None) and (firstPassageCDF > 1 / N):
                quantile = t

            t+=1
            nFirstPassageCDFPrev = nFirstPassageCDF

            if (time.time() - last_save_time) >= time_interval: 
                np.savetxt(f"PDF{sysID}.txt", pdf)
                np.savetxt(f"SaveDistance{sysID}.txt", [d])
                last_save_time = time.time()

        variance = running_sum_squared - running_sum ** 2
        writer.writerow([N, quantile, variance, d])
        f.flush()
        
    f.close()

if __name__ == "__main__":
    # Test line:
    (save_dir, sysID, distance, nMin, nMax, num_of_points) = sys.argv[1:]
    nMin = float(nMin)
    nMax = float(nMax)
    num_of_points = int(num_of_points)

    vars = {"nMin": nMin,
            "nMax": nMax,
            "distance": distance,
            "num_of_points": num_of_points,
            "save_dir": save_dir,
            "sysID": sysID}

    vars_file = os.path.join(save_dir, "variables.json")
    today = date.today()
    text_date = today.strftime("%b-%d-%Y")

    if int(sysID) == 0:
        vars.update({"Date": text_date})
        saveVars(vars, vars_file)
        vars.pop("Date")

    runExperiment(**vars)

