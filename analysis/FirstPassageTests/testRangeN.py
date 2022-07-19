import numpy as np 
import npquad
import sys 
sys.path.append("../../src")
from pyfirstPassagePDF import FirstPassagePDF

def sampleCDF(cdf, N):
    Ncdf = 1 - np.exp(-cdf.astype(float) * N)
    Npdf = np.diff(Ncdf)
    return Ncdf, Npdf


def calculateMeanAndVariance(x, pdf):
    mean = sum(x * pdf)
    var = sum(x ** 2 * pdf) - mean ** 2
    return mean, var

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    maxPosition = 500
    beta = np.inf
    Nvals = np.arange(2, 52, step=2)
    Nvals = [float(f"1e{i}") for i in Nvals]
    var_vals = []
    for N in Nvals:
        pdf = FirstPassagePDF(beta, maxPosition)
        data = pdf.evolveToCutoff(0.99, N)
        times = data[:, 0]
        pdf = data[:, 1]
        cdf = data[:, 2]
        Ncdf, Npdf = sampleCDF(cdf, N)
        mean_val, var_val = calculateMeanAndVariance(times[1:], Npdf)
        var_vals.append(var_val)

    fig, ax = plt.subplots()
    ax.plot(np.log(Nvals), var_vals)
    xvals = np.array([10, 100])
    yvals = 1/xvals**2.5 * 10**9
    ax.plot(xvals, yvals, c='k', ls='--', label=r'$(\log(N))^{-2.5}$')
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.set_xlabel(r'$\log(N)$')
    ax.set_ylabel(r"$\mathrm{Var}(\tau)$")
    fig.savefig("FirstPassageN.pdf", bbox_inches='tight')