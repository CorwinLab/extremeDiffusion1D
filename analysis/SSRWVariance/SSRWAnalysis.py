import numpy as np
from matplotlib import pyplot as plt 
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.special import erfc

def cumulativeDistribution(L, t, nTerms=50, verbose=False):
    sum = np.zeros(len(t))
    for k in range(0, nTerms):
        print('k=', k)
        # This is the correct way to do it but we can simplify
        # sum += (-1)**k * erfc(L * np.abs(1 + 2*k)/np.sqrt(2 * t)) * np.sign(1+2*k)
        sum += 2 * (-1) ** k * erfc(L*(1+2*k)/np.sqrt(2*t))
    return sum

def I(v): 
    return 1/2 * ((1+v)*np.log(1+v) + (1-v)*np.log(1-v))

def cumulativeDistributionSampling(L, t):
    return 2 * np.exp(- t * I(L/t))

def getNParticleMeanVar(positions, N, standardDeviations=50, numpoints=int(1e4), nTerms=50, verbose=False, theory='SSRW'):
    mean = np.zeros(len(positions))
    var = np.zeros(len(positions))
    for i in range(len(positions)):
        logN = np.log(N)
        L = positions[i]
        Nmean = L**2/2/logN
        Nvar = np.pi / 24 *L**4 / logN**4
        if theory == 'SSRW':
            tMin = 0.001
        else: 
            tMin = L + 1
        tMax = Nmean + standardDeviations * np.sqrt(Nvar)
        if tMin >= tMax:
            tMax = 100000*tMin
        t = np.linspace(tMin, tMax, numpoints)
        if theory == 'SSRW':
            single_particle_cdf = cumulativeDistribution(L, t, nTerms, verbose)
        elif theory == 'Sampling':
            single_particle_cdf = cumulativeDistributionSampling(L, t)
        nParticle_cdf = 1 - np.exp(-N * single_particle_cdf)
        pdf = np.diff(nParticle_cdf)
        if (sum(pdf)) < 0.9999999999999998:
            mean[i] = np.nan
            var[i] = np.nan
            continue
        mean[i] = np.sum(t[:-1] * pdf)
        var[i] = np.sum(t[:-1]**2 * pdf) - mean[i]**2
    return mean, var    

Nexps = [2, 5, 12, 28]

cm = LinearSegmentedColormap.from_list(
    "rg", ["tab:orange", "tab:red", "tab:purple", "tab:blue"], N=256
)
colors = [cm(1.0 * i / len(Nexps) / 1) for i in range(len(Nexps))]

fig, ax = plt.subplots()

for i, Nexp in enumerate(Nexps):
    df = pd.read_csv(f"MeanVar{Nexp}.txt")
    N = float(f"1e{Nexp}")
    ax.plot(df['Position'] / np.log(N), df['Mean'], c=colors[i], label=rf"$N=1e{Nexp}$")

xvals = np.array([100., 700.])
ax.plot(xvals, xvals ** 2, ls='--', c='k', label=r'$L^2$')
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim([0.5, 2*1e6])
ax.set_xlim([0.2, 750])
ax.set_xlabel(r"$L / \log(N)$")
ax.legend()
ax.set_ylabel(r"$\mathrm{Mean}(\tau_{\mathrm{Sam}})$")
fig.savefig("Mean.png", bbox_inches='tight')

fig, ax = plt.subplots()
for i, Nexp in enumerate(Nexps):
    df = pd.read_csv(f"MeanVar{Nexp}.txt")
    N = float(f"1e{Nexp}")
    ax.plot(df['Position'] / np.log(N), df['Variance'], c=colors[i], label=rf"$N=1e{Nexp}$")
    
xvals = np.array([100., 700.])
ax.plot(xvals, xvals ** 4 / 10, ls='--', c='k', label=r'$L^4$')
ax.set_xlabel(r"$L / \log(N)$")
ax.set_ylabel(r"$\mathrm{Var}(\tau_{\mathrm{Sam}})$")
ax.set_xlim([0.5, 750])
ax.set_ylim([1e-3, 1e12])
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend()
fig.savefig("Variance.png", bbox_inches='tight')

fig, ax = plt.subplots()
for i, Nexp in enumerate(Nexps):
    df = pd.read_csv(f"MeanVar{Nexp}.txt")
    N = float(f"1e{Nexp}")
    power_law = np.pi**2 / 24 * df['Position'].values ** 4 / np.log(N)**4
    power_law[df['Position'].values <= np.log(N)] = 0
    ax.plot(df['Position'] / np.log(N), df['Variance'] - power_law, c=colors[i], label=rf"$N=1e{Nexp}$")
    
xvals = np.array([100., 700.])
ax.plot(xvals, xvals ** 4 / 10, ls='--', c='k', label=r'$\pm L^4$')
ax.plot(xvals, -xvals ** 4, ls='--', c='k',)
ax.set_xlabel(r"$L / \log(N)$")
ax.set_ylabel(r"$\mathrm{Var}(\tau_{\mathrm{Sam}}) - \frac{\pi^2 L^4}{24 \log(N)^4}$")
ax.set_xlim([0.5, 750])
ax.set_xscale("log")
ax.set_yscale("symlog")
ax.legend()
fig.savefig("Residual.png", bbox_inches='tight')

fig, ax = plt.subplots()
for i, Nexp in enumerate(Nexps):
    df = pd.read_csv(f"MeanVar{Nexp}.txt")
    N = float(f"1e{Nexp}")
    mean, var = getNParticleMeanVar(df['Position'].values, N, nTerms=1)
    ax.plot(df['Position'] / np.log(N), -df['Variance']+var, c=colors[i], label=rf"$N=1e{Nexp}$")
    
xvals = np.array([10, 100])
ax.plot(xvals, xvals**2., c='k', ls='--', label=r'$L^2$')
ax.set_xlabel(r"$L / \log(N)$")
ax.set_ylabel(r"$-\mathrm{Var}^{\mathrm{Num}}(\tau_{\mathrm{Sam}}) + \mathrm{Var}^{\mathrm{Theory}}(\tau_{\mathrm{Sam}})$")
ax.set_xlim([0.5, 750])
ax.set_xscale("log")
ax.set_yscale("symlog")
ax.legend()
fig.savefig("TheoryResidual.png", bbox_inches='tight')

fig, ax = plt.subplots()
for i, Nexp in enumerate(Nexps):
    df = pd.read_csv(f"MeanVar{Nexp}.txt")
    N = float(f"1e{Nexp}")
    mean, var = getNParticleMeanVar(df['Position'].values, N, theory='Sampling')
    ax.plot(df['Position'] / np.log(N), df['Variance']-var, c=colors[i], label=rf"$N=1e{Nexp}$")
    
xvals = np.array([10, 100])
ax.plot(xvals, xvals**4., c='k', ls='--', label=r'$L^4$')
ax.set_xlabel(r"$L / \log(N)$")
ax.set_ylabel(r"$\mathrm{Var}^{\mathrm{Num}}(\tau_{\mathrm{Sam}}) - \mathrm{Var}^{\mathrm{Theory}}(\tau_{\mathrm{Sam}})$")
ax.set_xlim([0.5, 750])
ax.set_xscale("log")
ax.set_yscale("symlog")
ax.legend()
fig.savefig("SamplingResidual.pdf", bbox_inches='tight')

fig, ax = plt.subplots()
for i, Nexp in enumerate(Nexps):
    df = pd.read_csv(f"MeanVar{Nexp}.txt")
    N = float(f"1e{Nexp}")
    mean, var = getNParticleMeanVar(df['Position'].values, N, theory='Sampling')
    ax.plot(df['Position'] / np.log(N), df['Variance'], c=colors[i], label=rf"$N=1e{Nexp}$")
    ax.plot(df['Position'] / np.log(N), var, c=colors[i], ls='--')
    
xvals = np.array([10, 100])
ax.plot(xvals, xvals**4., c='k', ls='--', label=r'$L^4$')
ax.set_xlabel(r"$L / \log(N)$")
ax.set_ylabel(r"$\mathrm{Var}(\tau_{\mathrm{Sam}})$")
ax.set_xlim([0.5, 750])
ax.set_xscale("log")
ax.set_yscale("symlog")
ax.legend()
fig.savefig("Sampling.pdf", bbox_inches='tight')