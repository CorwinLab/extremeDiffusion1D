import numpy as np
import npquad
import sys
import os
from datetime import date
import csv
from pyDiffusion import FirstPassagePDF
from experimentUtils import saveVars

def runExperiment(beta, dmin, dmax, cutoff, N_exp, save_file):
    beta = float(beta)
    dmin = float(dmin)
    dmax = float(dmax)
    cutoff = float(cutoff)
    N = np.quad(f"1e{N_exp}")
    distances = np.geomspace(dmin, dmax, num=500).astype(int)
    distances = np.unique(distances)

    f = open(save_file, "a")
    writer = csv.writer(f)
    writer.writerow(['distance', 'var', 'quantile'])
    f.flush()
    for i, d in enumerate(distances):
        pdf = FirstPassagePDF(beta, d)
        quantile, variance, Ns = pdf.evolveToCutoffMultiple([N], cutoff)
        writer.writerow([d, variance[0], quantile[0]])
        f.flush()
        
    f.close()


if __name__ == "__main__":
    (topDir, beta, N_exp, sysID, dmin, dmax, cutoff) = sys.argv[1:]

    save_dir = f"{topDir}"
    save_file = os.path.join(save_dir, f"FirstPassageTimes{sysID}.txt")
    save_file = os.path.abspath(save_file)
    dmax = 500 * np.log(np.quad(f"1e{N_exp}")).astype(float)
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
