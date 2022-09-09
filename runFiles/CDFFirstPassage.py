import numpy as np
import npquad 
import sys 
import os 
from pyDiffusion import DiffusionTimeCDF
from experimentUtils import saveVars
from datetime import date

def runExperiment(tMax, N, distance, save_file):
    # choose uniform random distribution
    cdf = DiffusionTimeCDF('beta', [1, 1], tMax)
    cdf.evolveAndSaveFirstPassage(N, distance, save_file, tMax)

if __name__ == '__main__': 
    (
        topDir,
        sysID,
        distance,
        Nexp,
    ) = sys.argv[1:]
    
    distance = int(distance)
    N = np.quad(f"1e{Nexp}")
    save_file = os.path.join(topDir, f"FirstPassageTimes{sysID}.csv")

    # Set tMax to 4 * (expected first passage time)
    tMax = int(4 * (distance ** 2 / 2 / np.log(N))) 
    
    vars = {
        "tMax": tMax, 
        "N": N,
        "distance": distance,
        "save_file": save_file,
    }

    vars_file = os.path.join(topDir, "variables.json")
    today = date.today()
    text_date = today.strftime("%b-%d-%Y")

    if int(sysID) == 0:
        vars.update({"Date": text_date})
        saveVars(vars, vars_file)
        vars.pop("Date")

    runExperiment(**vars)
