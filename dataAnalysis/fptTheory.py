from scipy.special import erf 
from TracyWidom import TracyWidom
import numpy as np

def I(v):
    return 1 - np.sqrt(1-v**2)

def Iprime(v):
    return v / np.sqrt(1-v**2)

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
        prefactor = 1/(I(x / t_val) - x**2 / t_val ** 2 / np.sqrt(1 - (x/t_val)**2)) ** 2
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

def sam_variance_theory(x, N):
    #N = N/2
    t_vals = t0(x, N)
    beta = 1 / (I(x / t_vals) - x**2 / t_vals ** 2 / np.sqrt(1 - (x/t_vals)**2))
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
        prefactor = 1/(I(x / t_val) - x**2 / t_val ** 2 / np.sqrt(1 - (x/t_val)**2))
        mean.append(t_val)

    return mean

def mean_theory(x, N): 
    logN = np.log(N).astype(float)
    return np.piecewise(x, [x < logN, x > logN], [lambda x: x, lambda x: mean_longTime(x, N)])