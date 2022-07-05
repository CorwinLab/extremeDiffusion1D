import numpy as np
import npquad 
import sys 
import os
from datetime import date
import csv

src_path = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.append(src_path)

from pyfirstPassagePDF import FirstPassagePDF
from experimentUtils import saveVars

def sampleCDF(cdf, N):
    Ncdf = 1 - np.exp(-cdf * N)
    Npdf = np.diff(Ncdf)
    return Ncdf, Npdf

def calculateMeanAndVariance(x, pdf): 
    mean = sum(x*pdf)
    var = sum(x**2 * pdf) - mean ** 2
    return mean, var

def runExperiment(beta, dmin, dmax, cutoff, N_exp, save_file):
    N = np.quad(f"1e{N_exp}")
    distances = np.arange(dmin, dmax)

    f = open(save_file, "a")
    writer = csv.writer(f)
    writer.writerow(['distance', 'mean', 'var'])
    f.flush()

    for i, d in enumerate(distances): 
        pdf = FirstPassagePDF(beta, d)
        data = pdf.evolveToCutoff(cutoff)
        times = data[:, 0]
        pdf = data[:, 1]
        cdf = data[:, 2]
        nonzero_indeces = np.nonzero(pdf)
        Ncdf, Npdf = sampleCDF(cdf, N)
        mean_val, var_val = calculateMeanAndVariance(times[1:], Npdf)

        writer.writerow([d, mean_val, var_val])
        f.flush()

    f.close()

if __name__ == '__main__': 
    (
        topDir,
        beta,
        N_exp,
        sysID,
        dmin, 
        dmax, 
        cutoff
    ) = sys.argv[1:]

    save_dir = f"{topDir}"
    save_file = os.path.join(save_dir, f"FirstPassageTimes{sysID}.txt")
    save_file = os.path.abspath(save_file)

    vars = {
        "beta": beta,
        "N_exp": N_exp,
        "dmin": dmin, 
        "dmax": dmax, 
        "cutoff": cutoff
    }
    vars_file = os.path.join(save_dir, "variables.json")
    today = date.today()
    text_date = today.strftime("%b-%d-%Y")

    if int(sysID) == 0:
        vars.update({"Date": text_date})
        saveVars(vars, vars_file)
        vars.pop("Date")

    runExperiment(**vars)
