import numpy as np
import npquad
import theory
from scipy.optimize import minimize

def tgreater(t, N, x): 
    return np.abs(theory.quantileMean(N, t) + np.sqrt(theory.quantileVarShortTime(N, t)) - x)

def tlesser(t, N, x):
    return np.abs(theory.quantileMean(N, t) - np.sqrt(theory.quantileVarShortTime(N, t)) - x)

def variance(N, x):
    tvals = np.zeros(shape=(len(x), 2))
    logN = np.log(N).astype(float)
    for i, xval in enumerate(x):
        t0 = xval**2 / (logN)**(3/4) / 6
        tPlus = minimize(tgreater, t0, (N, xval), bounds=[(np.log(N).astype(float)+10, np.inf)])
        tMinus = minimize(tlesser, t0, (N, xval), bounds=[(np.log(N).astype(float)+10, np.inf)])
        tvals[i, 0] = tMinus.x
        tvals[i, 1] = tPlus.x
    return tvals
    
if __name__ == '__main__':
    from matplotlib import pyplot as plt
    N = np.quad(f"1e24")
    logN = np.log(N)
    x = np.geomspace(60, 6000)
    var = variance(N, x)
    fig, ax = plt.subplots()
    ax.plot(x, var[:, 0])
    ax.plot(x, var[:, 1])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    fig.savefig("Var.png")

    fig, ax = plt.subplots()
    ax.plot(x, var[:, 0] - var[:, 1])
    fig.savefig("VAr.png")