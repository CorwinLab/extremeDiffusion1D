import numpy as np
from theory import KPZ_var_fit

gamma = 0.577
tw_var = 0.813

def Iprime(r0, D, v):
    return v / 2 / D + 8 * r0**2 * v**3 / 3 / (8 * D)**3

def sigma(r0, D, v):
    return r0**(2/3) * v**(4/3) / 4 / D

def v0(r0, D, N, t):
    logN = np.log(N)
    c1 = 2 * r0**2 / 3 / (8 * D)**3
    c2 = 1 / 4 / D
    return 1/np.sqrt(2 * c1) * np.sqrt(np.sqrt(c2**2 + 4 * c1 * logN / t) - c2)

def sampling_mean(r0, D, N, t):
    v = v0(r0, D, N, t)
    return gamma / Iprime(r0, D, v)

def theoretical_mean(r0, D, N, t):
    return v0(r0, D, N, t) * t + sampling_mean(r0, D, N, t)

def environmental_variance(r0, D, N, t):
    v = v0(r0, D, N, t)
    return t**(2/3) * (sigma(r0, D, v) / Iprime(r0, D, v)) ** 2 * tw_var 

def theoretical_long_time_variance(r0, D, N, t):
    return D * t / np.log(N) * KPZ_var_fit(r0**2 * np.log(N)**2 / 4 / D / t) + sampling_variance(r0, D, N, t)

def sampling_variance(r0, D, N, t):
    v = v0(r0, D, N, t)
    return np.pi**2 / 6 / Iprime(r0, D, v)**2

def theoretical_variance(r0, D, N, t):
    return environmental_variance(r0, D, N, t) + sampling_variance(r0, D, N, t)