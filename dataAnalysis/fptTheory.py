from scipy.special import erf 
from TracyWidom import TracyWidom
import numpy as np
from numba import njit

def I(v):
    return 1 - np.sqrt(1-v**2)

def Iprime(x, t):
    return - x**2 / t**3 / np.sqrt(1-(x/t)**2)

def sigma(v):
    return (2 * I(v)**2 / (1-I(v)))**(1/3)

def t0(x, N):
    logN = np.log(N)
    return (x**2 + logN**2) / (2 * logN)

def function(x, N, chi1, chi2):
    t_vals = t0(x, N)
    exponent = t_vals**(1/3) * sigma(x / t_vals)
    return np.log(np.exp(exponent * chi1) + np.exp(exponent * chi2))

def var_power_long(x, N): 
    logN = np.log(N).astype(float)
    return 1/4 * np.sqrt(np.pi / 2) * x ** 3 / logN ** (5/2)

def var_power_short(x, N):
    logN = np.log(N).astype(float)
    return 0.8133 * x ** (8/3) / logN **(2) / 2 **(5/3)

def var_short(xvals, N, samples=10000):
    var = []
    tw = TracyWidom(beta=2)
    for x in xvals: 
        r1 = np.random.rand(samples)
        r2 = np.random.rand(samples)
        chi1 = tw.cdfinv(r1)
        chi2 = tw.cdfinv(r2)
        function_var = function(x, N, chi1, chi2)
        t_val = t0(x, N)
        prefactor = 1/(I(x / t_val) + t_val * Iprime(x, t_val)) ** 2
        var.append(prefactor * np.var(function_var))

    return var 

def variance(x, N, samples=10000):
    crossover = (np.log(N).astype(float)) ** (3/2)
    width = (np.log(N).astype(float))**(4/3)
    theory_short = var_short(x, N, samples)
    theory_long =  var_power_long(x, N)
    error_func = (erf((x - crossover) / width) + 1) / 2
    theory = theory_short * (1 - error_func) + theory_long * (error_func)
    theory[x < np.log(N)] = 0
    return theory

def sam_variance_theory(x, N, samples=10000):
    #N = N/2
    t_vals = t0(x, N)
    t_vals = np.array(t_vals)
    beta = 1 / (I(x / t_vals) + t_vals * Iprime(x, t_vals))
    beta[x < np.log(N)] = 0
    return np.pi**2 * beta ** 2 / 6

def mean_longTime(xvals, N, samples=10000): 
    mean = []
    tw = TracyWidom(beta=2)
    for x in xvals: 
        r1 = np.random.rand(samples)
        r2 = np.random.rand(samples)
        chi1 = tw.cdfinv(r1)
        chi2 = tw.cdfinv(r2)
        function_var = function(x, N, chi1, chi2)
        t_val = t0(x, N)
        prefactor = 1/(I(x / t_val) + t_val * Iprime(x, t_val))
        mean.append(t_val)

    return mean

def mean_theory(x, N): 
    logN = np.log(N).astype(float)
    return np.piecewise(x, [x < logN, x > logN], [lambda x: x, lambda x: t0(x, N)])

def numericalSamplingVariance(x, N):
    variance = np.zeros(len(x))
    t_mean = x**2/2/np.log(N)
    var = np.pi**2 / 6 * x **4 / np.log(N)**4
    tw_mean = -1.77

    for i in range(len(x)): 
        if var[i] < 0.5: 
            var[i] = 50
        tmin = x[i]
        tmax = int(t_mean[i] + 10 * np.sqrt(var[i]))
        t = np.arange(tmin, tmax, step=1)
        singleParticleCDF = np.exp(-t * (1-np.sqrt(1-(x[i]/t)**2)) + tw_mean * t**(1/3) * sigma(x[i]/t))
        cdf = 1 - np.exp(- 2* N * singleParticleCDF) #+ np.log(x[i]/t)) * 2 * np.exp(-x[i]**4 / t**3 / 24 - np.log(np.sqrt(2 * np.pi * x[i]**4 / t**3))))
        pdf = np.diff(cdf)
        mean = np.sum(t[1:] * pdf)
        variance[i] = np.sum(t[1:]**2 * pdf) - mean**2
    variance[x <= np.log(N)+1] = 0
    return variance

def linearInterpolation(x, xp, yp, yperr):
    y = np.zeros(shape=len(x))
    yerr = np.zeros(shape=len(x))
    for i, xval in enumerate(x): 
        idx = np.argmax(xp - xval >= 0)
        x2 = xp[idx]
        x1 = xp[idx-1]
        y2 = yp[idx]
        y1 = yp[idx-1]

        if x2 == xval: 
            y[i] = y2 
            yerr[i] = yperr[idx]
            continue 
        elif x1 == xval:
            y[i] = y1 
            yerr[i] = yperr[idx-1]
            continue 
        m =(y2 - y1)/(x2 - x1)
        b = y1 - m * x1 
        y[i] = m * xval + b
        yerr[i] = np.sqrt((xval-x1)**2/(x2-x1)**2 * yperr[idx] ** 2 + (xval-x2)**2/(x2-x1)**2 * yperr[idx - 1]**2)
        print(yerr[i] >= min([yperr[idx], yperr[idx-1]]))
    return y, yerr
    
if __name__ == '__main__':
    from matplotlib import pyplot as plt

    x = np.linspace(0.5, 5.5, num=7)
    xp = np.linspace(-1, 6, num=50)
    yp = xp**4
    yperr = np.sqrt(np.abs(xp**4 * np.random.normal(0, xp**2, size=len(xp))))
    y, yerr = linearInterpolation(x, xp, yp, yperr)
    print(yerr)
    fig, ax = plt.subplots()
    ax.errorbar(xp, yp, yperr, fmt='o', ms=3, alpha=0.5)
    ax.errorbar(x, y, yerr, fmt='o', ms=3, color='r')
    fig.savefig("InterpTest.png", bbox_inches='tight')