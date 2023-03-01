import numpy as np
from scipy.special import lambertw, erfc, zeta
from matplotlib import pyplot as plt
import sys 
import pandas as pd
sys.path.append("./dataAnalysis")
from fptTheory import sam_variance_theory

def cumulativeDistribution(L, t, nTerms=50, verbose=False):
    sum = np.zeros(len(t))
    for k in range(0, nTerms):
        # This is the correct way to do it but we can simplify
        # sum += (-1)**k * erfc(L * np.abs(1 + 2*k)/np.sqrt(2 * t)) * np.sign(1+2*k)
        if verbose:
            print(f'k={k}')
        sum += 2 * (-1) ** k * erfc(L*(1+2*k)/np.sqrt(2*t))
    return sum

def getNParticleMeanVar(positions, N, standarDeviations=10, numpoints=int(1e4), nTerms=50, verbose=False):
    mean = np.zeros(len(positions))
    var = np.zeros(len(positions))
    for i in range(len(positions)):
        logN = np.log(N)
        L = positions[i]
        Nmean = L**2/2/logN
        Nvar = np.pi / 24 *L**4 / logN**4
        tMin = 1
        tMax = Nmean + standarDeviations * np.sqrt(Nvar)
        if tMin < 0:
            tMin = 0
        t = np.linspace(tMin, tMax, numpoints)
        single_particle_cdf = cumulativeDistribution(L, t, nTerms, verbose)
        nParticle_cdf = 1 - np.exp(-N * single_particle_cdf)
        pdf = np.diff(nParticle_cdf)

        mean[i] = np.sum(t[:-1] * pdf)
        var[i] = np.sum(t[:-1]**2 * pdf) - mean[i]**2
    return mean, var

def moment_function(L, N, m):
    gamma = np.euler_gamma
    beta = 2 * np.log(N) / L**2
    C = L**2 / 2
    D1 = - 1/L**2
    theta= 1/2 
    return beta ** m * (1 + m*gamma/C/beta + (m*(m-1)*(6*gamma**2 + np.pi**2) - 12 * m * theta * gamma + 12 * m * C * D1)/(12 * C**2 * beta**2))

def mean_function_squared(L, N):
    gamma = np.euler_gamma
    beta =  2 * np.log(N) / L**2
    C = L**2 / 2
    D1 = - 1/L**2
    theta= 1/2 
    m = -1
    return beta ** -2 * (1 + 2 * m * gamma / C / beta + 2*(m*(m-1)*(6*gamma**2 + np.pi**2) - 12 * m * theta * gamma + 12 * m * C * D1)/(12 * C**2 * beta**2) 
                        + (m * gamma / C / beta)**2)

def moment_generating_function(N, C, D0, D1, D2, theta, m):
    gamma = np.euler_gamma
    beta = theta / C * np.log(C / theta * (D0*N)**(1/theta)) # approximation of w-lambert function
    bn = beta * (1 + D1 / C / beta**2 - (2 * theta * D1 + C*(D1**2 - 2*D2))/(2 * C**2 * beta**3) )
    an = 1/C * (1- theta/C/beta + (theta**2-C*D1) / C**2 / beta**2 - (2*theta**3 - 6 * theta * C * D1 - 2 * C**2 *(D1**2 - 2 * D2))/2/C**3 /beta**3)
    c2 = -theta / np.log(N)**2 
    c3 = 2 * theta / np.log(N)**3

    second_order = -m * (-an / bn) / 12 *(-6  * c2 * gamma**2 + gamma**3 *(6 * c2 **2 - 2 * c3) - c2 * np.pi**2 + gamma*(12 + (3 * c2**2-c3)*np.pi**2) + 12 * c2**2 * zeta(3) - 4 * c3 * zeta(3)) 

    return bn**m * (1 + second_order)

N = 1e12
L = np.geomspace(750*np.log(N), 1000*np.log(N), 1000)

Ns = np.geomspace(10, 1e24)
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.plot(np.log(Ns), 2 * np.log(Ns), label='Asymptotics')
ax.plot(np.log(Ns), lambertw(2 / np.pi * Ns**2), label='W-Lambert')
ax.legend()
fig.savefig("LmabertTest.png", bbox_inches='tight')

var = moment_function(L, N, -2) - mean_function_squared(L, N)
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.plot(L/np.log(N), var / L**4, label='Arxiv Paper')
ax.plot(L/np.log(N), np.pi**2 * L**4 / 24 /np.log(N)**4/L**4, ls='-.', label=r'$\frac{\pi^2 L^4}{24 \log(N)^4}$')
ax.plot(L/np.log(N), sam_variance_theory(L, N)/L**4, ls='--', label='Our Theory')
ax.legend()
fig.savefig("TestVar.png",bbox_inches='tight')

einstein_df = pd.read_csv('/home/jacob/Desktop/talapasMount/JacobData/FPTDiscreteE/12/MeanVariance.csv')
einstein_df = einstein_df[einstein_df['Distance'] > 5*10**3]
L = einstein_df['Distance'].values

var = moment_function(L, N, -2) - mean_function_squared(L, N)

theoretical_mean, theoretical_var = getNParticleMeanVar(L, N)

var_higher_order = moment_generating_function(N, C = L**2/2, D0 = np.sqrt(2/np.pi/L**2), D1=-1/L**2, D2=3/L**4, theta=1/2, m=-2) - mean_function_squared(L, N)

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.plot(einstein_df['Distance'], einstein_df['Variance'] / L**4, label='Einstein Data')
ax.plot(L, sam_variance_theory(L, N) / L**4, ls='--', label='Our Theory')
ax.plot(L, np.pi**2 * L**4 / 24 /np.log(N)**4/L**4, ls='-.', label=r'$\frac{\pi^2 L^4}{24 \log(N)^4}$')
ax.plot(L, var / L**4, label='Arxiv Paper')
ax.plot(L, theoretical_var / L**4, ls='--', label='True Theory')
ax.legend(loc='lower right')
fig.savefig("EinstenVar.png", bbox_inches='tight')