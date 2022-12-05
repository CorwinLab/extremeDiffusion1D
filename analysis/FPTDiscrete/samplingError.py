import numpy as np
import sys
from matplotlib import pyplot as plt 
import os 
import pandas as pd
sys.path.append("../../dataAnalysis")
from fptTheory import sam_variance_theory, numericalSamplingVariance
from theory import log_moving_average
from scipy.special import erfc
from matplotlib.colors import LinearSegmentedColormap

def cumulativeDistribution(L, t, nTerms=50, verbose=False):
    sum = np.zeros(len(t))
    for k in range(0, nTerms):
        # This is the correct way to do it but we can simplify to only positive k
        # sum += (-1)**k * erfc(L * np.abs(1 + 2*k)/np.sqrt(2 * t)) * np.sign(1+2*k)
        if verbose:
            print(f'k={k}')
        sum += 2 * (-1) ** k * erfc(L*(1+2*k)/np.sqrt(2*t))
        #x = L * (1+2*k) / np.sqrt(2*t)
        #sum += 2 * (-1) ** k * np.exp(-x**2)/x/np.sqrt(np.pi)
    return sum

def getNParticleMeanVar(positions, N, standarDeviations=10, numpoints=int(1e4), nTerms=50, verbose=False):
    mean = np.zeros(len(positions))
    var = np.zeros(len(positions))
    for i in range(len(positions)):
        logN = np.log(N)
        L = positions[i]
        Nmean = L**2/2/logN
        Nvar = np.pi / 24 *L**4 / logN**4
        tMin = 0.001
        tMax = Nmean + standarDeviations * np.sqrt(Nvar)
        t = np.linspace(tMin, tMax, numpoints)
        single_particle_cdf = cumulativeDistribution(L, t, nTerms, verbose)
        nParticle_cdf = 1 - np.exp(-N * single_particle_cdf)
        pdf = np.diff(nParticle_cdf)
        mean[i] = np.sum(t[:-1] * pdf)
        var[i] = np.sum(t[:-1]**2 * pdf) - mean[i]**2
    return mean, var

def getNParticleMeanVarTW(positions, N, standarDeviations=10, numpoints=int(1e4), nTerms=50, verbose=False):
    mean = np.zeros(len(positions))
    var = np.zeros(len(positions))
    for i in range(len(positions)):
        logN = np.log(N)
        L = positions[i]
        Nmean = L**2/2/logN
        Nvar = np.pi / 24 *L**4 / logN**4
        tMin = L
        tMax = Nmean + standarDeviations * np.sqrt(Nvar)
        t = np.linspace(tMin, tMax, numpoints)
        single_particle_cdf = 2 * np.exp(-L**2 / 2 / t - L**4 / 8 / t**3 + np.log(L/t) - np.log(np.sqrt(2*np.pi*L**4/t**3)))
        nParticle_cdf = 1 - np.exp(-N * single_particle_cdf)
        pdf = np.diff(nParticle_cdf)
        mean[i] = np.sum(t[:-1] * pdf)
        var[i] = np.sum(t[:-1]**2 * pdf) - mean[i]**2
    return mean, var

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("symlog")
ax.set_ylabel(r'$\mathrm{Var}(\tau_{\mathrm{Sam}}) - \mathrm{Var}^{\mathrm{Theory}}(\tau_{\mathrm{Sam}})$')
ax.set_xlabel(r"$L / \log(N)$")
Nlabels = [r'$N=10$', r'$N=10^2$', r'$N=10^{5}$', r'$N=10^{12}$', r'$N=10^{28}$']

cm = LinearSegmentedColormap.from_list(
    "rg", ["tab:orange", "tab:red", "tab:purple", "tab:blue"], N=256
)
colors = [cm(1.0 * i / len(Nlabels) / 1) for i in range(len(Nlabels))]
alpha = 0.6

for i, Nexp in enumerate([1, 2, 5, 12, 28]):
    cdf_dir = f'/home/jacob/Desktop/corwinLabMount/CleanData/FPTCDFPaper/{Nexp}/'
    cdf_file = os.path.join(cdf_dir, 'MeanVariance.csv')
    cdf_df = pd.read_csv(cdf_file)

    N = float(f"1e{Nexp}")
    logN = np.log(N)
    
    if Nexp in [12, 28]:
        theory = sam_variance_theory(cdf_df['Distance'], N)
        ax.plot(cdf_df['Distance'] / logN, cdf_df['Sampling Variance'] - theory, label=Nlabels[i] + "(Gumbel Approximation)", c=colors[i], ls='--')
    '''
    home_dir = f'/home/jacob/Desktop/talapasMount/JacobData/FPTDiscretePaper/{Nexp}/'
    max_file = os.path.join(home_dir, 'MeanVariance.csv')
    max_df = pd.read_csv(max_file)
    mean, var = getNParticleMeanVar(max_df['Distance'], N, nTerms=1)
    if Nexp not in [28]:
        ax2.plot(max_df['Distance'] / logN, max_df['Variance'] - var, c=colors[i], alpha=0.75, label=Nlabels[i])
    '''
    mean, var = getNParticleMeanVarTW(cdf_df['Distance'], N, nTerms=1)
    residual = cdf_df['Sampling Variance'] - var
    ax.plot(cdf_df['Distance'] / logN, residual, label=Nlabels[i], c=colors[i], alpha=0.75)

xvals = np.array([100, 600])
ax.plot(xvals, xvals**(4), ls='--', c='k', label=r'$L^{4}$')

''' SSRW Analysis
einstein_df = pd.read_csv('/home/jacob/Desktop/talapasMount/JacobData/FPTDiscreteE/12/MeanVariance.csv')
new_row = [np.log2(N), 0]
einstein_theoretical_data = np.loadtxt("etheoretical12.txt")
einstein_theoretical_data = np.vstack([new_row, einstein_theoretical_data])

einstein_theoretical_data = np.interp(einstein_df['Distance'].values, einstein_theoretical_data[:, 0], einstein_theoretical_data[:, 1])

ax.plot(einstein_df['Distance'] / np.log(1e12), einstein_df['Variance'] - einstein_theoretical_data)
'''

ax.set_xlim([0.5, 750])
labels = [float(f"1e{i}") for i in range(1, 12, 2)]
neglabels = [-float(f"1e{i}") for i in range(1, 12, 2)]
labels = neglabels + [0] + labels
ax.set_yticks(labels)
#ax.legend(framealpha=0)
fig.savefig("SamplingError.pdf", bbox_inches='tight')
