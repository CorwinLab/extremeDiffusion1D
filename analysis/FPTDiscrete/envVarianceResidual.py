import numpy as np
import os 
import pandas as pd
from matplotlib import pyplot as plt
import sys 
sys.path.append("../../dataAnalysis")
from fptTheory import variance
from matplotlib.colors import LinearSegmentedColormap
from scipy.special import erfc
from theory import log_moving_average
plt.rcParams.update({'font.size': 12})

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
        tMin = 0.001
        tMax = Nmean + standarDeviations * np.sqrt(Nvar)
        t = np.linspace(tMin, tMax, numpoints)
        single_particle_cdf = cumulativeDistribution(L, t, nTerms, verbose)
        nParticle_cdf = 1 - np.exp(-N * single_particle_cdf)
        pdf = np.diff(nParticle_cdf)
        mean[i] = np.sum(t[:-1] * pdf)
        var[i] = np.sum(t[:-1]**2 * pdf) - mean[i]**2
    return mean, var

Nlabels = [r'$N=10$', r'$N=10^2$', r'$N=10^{5}$', r'$N=10^{12}$', r'$N=10^{28}$']
Ns = [1, 2, 5, 12, 28]
max_dists = [1725, 3452, 8630, 20721, np.log(1e28) * 750]

cm = LinearSegmentedColormap.from_list(
    "rg", ["tab:orange", "tab:red", "tab:purple", "tab:blue"], N=256
)
colors = [cm(1.0 * i / len(Nlabels) / 1) for i in range(len(Nlabels))]
alpha = 0.6

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("symlog")
ax.set_xlabel(r"$L / \log(N)$")
ax.set_ylabel(r"$\mathrm{Var}^{\mathrm{Num}}(\tau_{\mathrm{Min}})-\mathrm{Var}^{\mathrm{Num}}(\tau_{\mathrm{Sam}})$")

for i, Nexp in enumerate(Ns):
    if Nexp < 5:
        continue
    cdf_dir_specific = f'/home/jacob/Desktop/corwinLabMount/CleanData/FPTCDFPaper/{Nexp}'
    dir = f'/home/jacob/Desktop/talapasMount/JacobData/FPTDiscretePaper/{Nexp}/'

    N = float(f"1e{Nexp}")
    logN = np.log(N)
    cdf_file = os.path.join(cdf_dir_specific, 'MeanVariance.csv')
    cdf_df = pd.read_csv(cdf_file)
    var_theory = variance(cdf_df['Distance'].values, N)
    ax.plot(cdf_df['Distance'] / logN, var_theory, c=colors[i], ls='--', alpha=0.5)
    #ax.plot(cdf_df['Distance'] / logN, cdf_df['Env Variance'], label=Nlabels[i], c=colors[i], alpha=alpha)
    
    max_file = os.path.join(dir, 'MeanVariance.csv')
    max_df = pd.read_csv(max_file)
    decade_scaling = 25
    max_at_cdf = np.interp(cdf_df['Distance'].values, max_df['Distance'].values, max_df['Variance'])
    dist_new, env_measured = log_moving_average(cdf_df['Distance'].values, max_at_cdf - cdf_df['Sampling Variance'], 10**(1/decade_scaling))
    ax.scatter(dist_new / logN, env_measured, label=Nlabels[i], color=colors[i], s=1)

xvals = np.array([10, 600])
ax.plot(xvals, 10 * xvals ** 3, c='k', ls='--', label=r'$L^3$')

leg = ax.legend(
    loc="upper left",
    framealpha=0,
    labelcolor=['k'] + colors[2:],
    handlelength=0,
    handletextpad=0,
)
#ax.set_ylim([10**-3, 10**12])
ax.set_xlim([0.5, 750])
for item in leg.legendHandles:
    item.set_visible(False)

fig.savefig("EnvironmentalVarianceResidual.pdf", bbox_inches='tight')