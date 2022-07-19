from distutils.sysconfig import customize_compiler
import numpy as np
import npquad 
import sys 
sys.path.append("../../src")
from pyfirstPassagePDF import FirstPassagePDF
import csv

def sampleCDF(cdf, N):
    Ncdf = 1 - np.exp(-cdf * N)
    Npdf = np.diff(Ncdf)
    return Ncdf, Npdf


def calculateMeanAndVariance(x, pdf):
    mean = sum(x * pdf)
    var = sum(x ** 2 * pdf) - mean ** 2
    return mean, var


def runExperiment(beta, dmin, dmax, cutoff, N_exp, save_file):
    beta = float(beta)
    dmin = float(dmin)
    dmax = float(dmax)
    cutoff = float(cutoff)
    N = np.quad(f"1e{N_exp}")
    distances = np.arange(dmin, dmax).astype(int)

    f = open(save_file, "a")
    writer = csv.writer(f)
    writer.writerow(['distance', 'mean', 'var', 'quantile'])

    for i, d in enumerate(distances):
        pdf = FirstPassagePDF(beta, d)
        data = pdf.evolveToCutoff(cutoff, N)
        times = data[:, 0]
        pdf = data[:, 1]
        cdf = data[:, 2]
        quantile = np.argmax(cdf > 1 / N)
        quantile_time = times[quantile]
        Ncdf, Npdf = sampleCDF(cdf, N)
        mean_val, var_val = calculateMeanAndVariance(times[1:], Npdf)

        writer.writerow([d, mean_val, var_val, quantile_time])
        f.flush()

    f.close()

if __name__ == '__main__':
    beta = 1
    dmin = 50
    dmax = 250
    cutoff=1
    N_exp = 24
    save_file = 'data.txt'
    runExperiment(beta, dmin, dmax, cutoff, N_exp, save_file)