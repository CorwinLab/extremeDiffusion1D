import numpy as np
import npquad
import sys
import os
from datetime import date
import csv

from pyDiffusion import FirstPassagePDF
from experimentUtils import saveVars

def runExperiment(beta, d, N_min, N_max, number_of_Ns, save_file,  cutoff=1):
    beta = float(beta)
    d = int(d)
    cutoff=float(cutoff)
    N_min = int(N_min)
    N_max = int(N_max)
    number_of_Ns = int(number_of_Ns)

    f = open(save_file, "a")
    writer = csv.writer(f)
    writer.writerow(['N', 'var', 'quantile'])
    Ns = np.unique(np.geomspace(N_min, N_max, number_of_Ns).astype(int))

    Ns = [np.quad(f"1e{N_exp}") for N_exp in Ns]
    pdf = FirstPassagePDF(beta, d)
    quantile, variance, Ns = pdf.evolveToCutoffMultiple(Ns, cutoff)
    for q, v, N in zip(quantile, variance, Ns):
        writer.writerow([N, v, q])
    
    f.flush()
    f.close()


if __name__ == "__main__":
    (topDir, sysID, beta, d, cutoff, N_min, N_max, number_of_Ns) = sys.argv[1:]

    save_dir = f"{topDir}"
    save_file = os.path.join(save_dir, f"FirstPassageTimes{sysID}.txt")
    save_file = os.path.abspath(save_file)

    vars = {
        "beta": beta,
        "d": d,
        "cutoff": cutoff,
        "N_min": N_min,
        "N_max": N_max,
        "number_of_Ns": number_of_Ns,
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
