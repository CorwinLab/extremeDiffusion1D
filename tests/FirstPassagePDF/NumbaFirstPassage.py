import numpy as np
from pyDiffusion import pyfirstPassageNumba
import os
import csv
import sys


def runExperiment(nExp, dMin, dMax, num_of_points, save_dir, sysID):
    N = float(f"1e{nExp}")
    dMin = int(dMin * np.log(N)) + 1
    dMax = int(dMax * np.log(N))
    distances = np.geomspace(dMin, dMax, num_of_points)
    distances = np.unique(distances.astype(int))

    save_file = os.path.join(save_dir, f"FirstPassageCDF{sysID}.txt")
    f = open(save_file, "a")
    writer = csv.writer(f)
    writer.writerow(["Position", "Quantile", "Variance"])

    for d in distances:
        quantile = None
        running_sum_squared = 0
        running_sum = 0

        pdf = pyfirstPassageNumba.initializePDF(d)
        firstPassageCDF = pdf[0] + pdf[-1]
        nFirstPassageCDFPrev = 1 - np.exp(-firstPassageCDF * N)
        t = 1

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
        variance = running_sum_squared - running_sum ** 2
        writer.writerow([d, quantile, variance])
        f.flush()
    f.close()

if __name__ == "__main__":
    # Test line:
    save_dir, sysID, dMin, dMax, nExp, num_of_points = '.', 1, 0, 50, 24, 250
    #(save_dir, sysID, dMin, dMax, nExp, num_of_points) = sys.argv[1:]
    dMin = float(dMin)
    dMax = float(dMax)
    num_of_points = int(num_of_points)

    vars = {"nExp": nExp,
            "dMin": dMin,
            "dMax": dMax,
            "num_of_points": num_of_points,
            "save_dir": save_dir,
            "sysID": sysID}

    runExperiment(**vars)

