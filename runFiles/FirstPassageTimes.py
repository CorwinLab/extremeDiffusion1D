import numpy as np
import npquad
import sys
import os
from datetime import date
from pyDiffusion import FirstPassageDriver
from experimentUtils import saveVars

def runExperiment(beta, dmin, dmax, cutoff, N_exp, save_file):
    beta = float(beta)
    dmin = float(dmin)
    dmax = float(dmax)
    cutoff = float(cutoff)
    N = np.quad(f"1e{N_exp}")
    distances = np.geomspace(dmin, dmax, num=500).astype(int)
    distances = np.unique(distances)

    if os.path.isfile(save_file) and os.stat(save_file).st_size != 0:
        data = np.loadtxt(save_file, skiprows=1, delimiter=',')
        maxDistance = data[-1, 0]
        distances = distances[distances > maxDistance]
        write_header = False

    else:
        write_header = True

    pdf = FirstPassageDriver(beta, distances)
    _ = pdf.evolveToCutoff(N, save_file, cutoff, write_header)

if __name__ == "__main__":
    # Testing line 
    topDir = '.'; beta=1; N_exp=24; sysID=0; dmin=10; dmax=500; cutoff=1; 
    #(topDir, beta, N_exp, sysID, dmin, dmax, cutoff) = sys.argv[1:]

    save_dir = f"{topDir}"
    save_file = os.path.join(save_dir, f"FirstPassageTimes{sysID}.txt")
    save_file = os.path.abspath(save_file)
    dmax = 500*np.log(np.quad(f"1e{N_exp}")).astype(float)
    vars = {
        "beta": beta,
        "N_exp": N_exp,
        "dmin": dmin, 
        "dmax": dmax, 
        "cutoff": cutoff,
        "save_file": save_file
    }

    vars_file = os.path.join(save_dir, "variables.json")
    today = date.today()
    text_date = today.strftime("%b-%d-%Y")

    if int(sysID) == 0:
        vars.update({"Date": text_date})
        saveVars(vars, vars_file)
        vars.pop("Date")

    runExperiment(**vars)
