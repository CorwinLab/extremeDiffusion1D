import numpy as np
from pyDiffusion import pyfirstPassageNumba
import os
import csv
import sys
from datetime import date
from experimentUtils import saveVars

def runExperiment(nExp, dMin, dMax, num_of_points, save_dir, sysID):
    N = float(f"1e{nExp}")
    dMin = int(dMin * np.log(N))
    dMax = int(dMax * np.log(N))
    distances = np.geomspace(dMin, dMax, num_of_points)
    distances = np.unique(distances.astype(int))

    for d in distances:
        save_file = os.path.join(save_dir, f"{sysID}FirstPassageCDF{d}.txt")
        f = open(save_file, "a")
        writer = csv.writer(f)
        writer.writerow(["time", "SingleParticleCDF"])
        f.flush()

        pdf = pyfirstPassageNumba.initializePDF(d)
        firstPassageCDF = pdf[0] + pdf[-1]
        nFirstPassageCDF = 1 - np.exp(-firstPassageCDF * N)
        t = 1
        while (nFirstPassageCDF < 1) or (firstPassageCDF < 1 / N):
            pdf = pyfirstPassageNumba.iteratePDF(pdf)
            firstPassageCDF = pdf[0] + pdf[-1]
            nFirstPassageCDF = 1 - np.exp(-firstPassageCDF * N)
            writer.writerow([t, firstPassageCDF])
            f.flush()
            t+=1

        f.close()

if __name__ == "__main__":
    (save_dir, sysID, dMin, dMax, nExp, num_of_points) = sys.argv[1:]
    dMin = float(dMin)
    dMax = float(dMax)
    num_of_points = int(num_of_points)

    vars = {"nExp": nExp,
            "dMin": dMin,
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

