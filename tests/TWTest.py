import numpy as np
import npquad 
from matplotlib import pyplot as plt
from TracyWidom import TracyWidom

tw = TracyWidom(beta=2)
r = np.random.rand(10000)
tw_sample = tw.cdfinv(r)
print(f"Mean: {np.mean(tw_sample)}")
print(f"Var: {np.var(tw_sample)}")

fig, ax = plt.subplots()
ax.hist(tw_sample, bins=50)
fig.savefig("TWDistribution.png", bbox_inches='tight')

def I(v):
    return 1 - np.sqrt(1-v**2)

def sigma(v):
    return (2 * I(v)**2 / (1-I(v)))**(1/3)

def t0(x, N):
    logN = np.log(N)
    return (x**2 + logN**2) / (2 * logN)

def function(x, N, chi1, chi2):
    t_vals = t0(x, N)
    exponent = t_vals**(1/3) * sigma(x / t_vals)
    return np.log(np.exp(exponent * chi1) + np.exp(exponent * chi2))

def variance(xvals, N):
    var = []
    for x in xvals: 
        r1 = np.random.rand(10000)
        r2 = np.random.rand(10000)
        chi1 = tw.cdfinv(r1)
        chi2 = tw.cdfinv(r2)
        function_var = function(x, N, chi1, chi2)
        t_val = t0(x, N)
        prefactor = 1/(I(x / t_val) - x**2 / t_val ** 2 / np.sqrt(1 - (x/t_val)**2)) ** 2
        var.append(prefactor * np.var(function_var))

    return var 


r1 = np.random.rand(10000)
r2 = np.random.rand(10000)
tw_sample1 = tw.cdfinv(r1)
tw_sample2 = tw.cdfinv(r2)

x = 100
N = 1e7
yvals = function(x, N, tw_sample1, tw_sample2)

fig, ax = plt.subplots()
ax.hist(yvals, bins=50)
fig.savefig("FunctionDistribution.png", bbox_inches='tight')

N = 1e7
x = np.geomspace(np.log(N), 1000 * np.log(N), num=500)
logN = np.log(N)

def var_short(x, N): 
    logN = np.log(N)
    return 1/2**(5/3) * x**(8/3) / logN**2 * 0.8133

var = variance(x, N)
fig, ax = plt.subplots()
ax.plot(x / logN, var)
ax.plot(x / logN, var_short(x, N))
ax.set_xscale("log")
ax.set_yscale("log")
fig.savefig("Variance.png", bbox_inches='tight')