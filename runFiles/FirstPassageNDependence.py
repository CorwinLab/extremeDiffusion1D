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
    mean = sum(x * pdf)
    var = sum(x ** 2 * pdf) - mean ** 2
    return mean, var

def runExperiment(beta, d, cutoff, N_min, N_max, number_of_Ns, save_file):
    beta = float(beta)
    d = int(d)
    cutoff=float(cutoff)
    N_min = int(N_min)
    N_max = int(N_max)
    number_of_Ns = int(number_of_Ns)

    f = open(save_file, "a")
    writer = csv.writer(f)
    writer.writerow(['distance', 'mean', 'var', 'quantile'])
    Ns = np.unique(np.geomspace(N_min, N_max, number_of_Ns).astype(int))

    for i, N_exp in enumerate(Ns):
        N = np.quad(f"1e{N_exp}")
        pdf = FirstPassagePDF(beta, d)
        data = pdf.evolveToCutoff(cutoff, N)
        times = data[:, 0]
        pdf = data[:, 1]
        cdf = data[:, 2]
        quantile = np.argmax(cdf > 1 / N)
        quantile_time = times[quantile]
        Ncdf, Npdf = sampleCDF(cdf, N)
        mean_val, var_val = calculateMeanAndVariance(times[1:], Npdf)

        writer.writerow([N_exp, mean_val, var_val, quantile_time])
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
