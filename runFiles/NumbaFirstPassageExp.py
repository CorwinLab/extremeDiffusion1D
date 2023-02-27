import numpy as np
from pyDiffusion import pyfirstPassageNumba
import os
import csv
import sys
from datetime import date
import time
from experimentUtils import saveVars


def runExperiment(nExp, dMax, num_of_points, save_dir, sysID):
    N = float(f"1e{nExp}")
    dMax = int(dMax * np.log(N))
    distances = np.geomspace(1, dMax, num_of_points)
    distances = np.unique(distances.astype(int))
    distances = distances[distances <= 750 * np.log(N)]

    write_header = True
    save_file = os.path.join(save_dir, f"FirstPassageCDF{sysID}.txt")
    if os.path.exists(save_file):
        data = np.loadtxt(save_file, skiprows=1, delimiter=',')
        max_position = data[-1, 0]
        if max_position == max(distances):
            sys.exit()
        distances = distances[distances > max_position]
        write_header = False
    
    f = open(save_file, "a")
    writer = csv.writer(f)
    if write_header:
        writer.writerow(["Position", "Quantile", "Variance"])
        f.flush()
    
    time_interval = 3600 * 12

    for d in distances:
        quantile = None
        running_sum_squared = 0
        running_sum = 0

        pdf = pyfirstPassageNumba.initializePDF(d)
        firstPassageCDF = pdf[0] + pdf[-1]
        nFirstPassageCDFPrev = 1 - np.exp(-N * firstPassageCDF)
        
        t = 1
        last_save_time = time.time()
        while (nFirstPassageCDFPrev < 1) or (firstPassageCDF < 1 / N):
            pdf = pyfirstPassageNumba.iteratePDF(pdf)

            firstPassageCDF = pdf[0] + pdf[-1]
            nFirstPassageCDF = 1 - np.exp(-N * firstPassageCDF)
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
        writer.writerow([d, quantile, variance])
        f.flush()
        
    f.close()

if __name__ == "__main__":
    # Test line:
    # save_dir, sysID, dMin, dMax, nExp, num_of_points = '.', 1, 0, 50, 24, 250
    (save_dir, sysID, dMax, nExp, num_of_points) = sys.argv[1:]
    dMax = float(dMax)
    num_of_points = int(num_of_points)

    vars = {"nExp": nExp,
            "dMax": dMax,
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

