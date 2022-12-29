import numpy as np
import glob 
from matplotlib import pyplot as plt
import pandas as pd
from scipy.special import erfc

gamma = 0.577 

def cumulativeDistribution(L, t, nTerms=50, verbose=False):
    sum = np.zeros(len(t))
    for k in range(0, nTerms):
        # This is the correct way to do it but we can simplify
        # sum += (-1)**k * erfc(L * np.abs(1 + 2*k)/np.sqrt(2 * t)) * np.sign(1+2*k)
        if verbose:
            print(f'k={k}')
        sum += 2 * (-1) ** k * erfc(L*(1+2*k)/np.sqrt(2*t))
    return sum

def getNParticlePDF(L, N, standarDeviations=10, numpoints=int(1e4), nTerms=50, verbose=False):
    logN = np.log(N)
    Nmean = L**2/2/logN
    Nvar = np.pi / 24 *L**4 / logN**4
    tMin = 0.001
    tMax = Nmean + standarDeviations * np.sqrt(Nvar)
    if tMin < 0:
        tMin = 0
    t = np.linspace(tMin, tMax, numpoints)
    single_particle_cdf = cumulativeDistribution(L, t, nTerms, verbose)
    nParticle_cdf = 1 - np.exp(-N * single_particle_cdf)
    pdf = np.diff(nParticle_cdf)
    return t[1:], pdf

def gumbelPDF(x, mean, var):
    beta = -np.sqrt(6 * var / np.pi**2)
    mu = mean - beta*gamma
    z = (x - mu) / beta 
    return 1/np.abs(beta) * np.exp(-(z + np.exp(-z)))

runAgain = False

if runAgain:
    maxDistance = 20721
    dir = '/home/jacob/Desktop/talapasMount/JacobData/FPTDiscretePaper/12/Q*.txt'
    files = glob.glob(dir)
    times = []
    i = 0
    for f in files:
        data = pd.read_csv(f)
        try:
            time = data[data['Distance'] == maxDistance]['Time'].values[0]
        except:
            continue
        times.append(time)
        i+=1
        print(i / len(files)* 100, '%')
    np.savetxt("Times.txt", times)

else:
    times = np.loadtxt("Times.txt")
    maxDistance = 20721

N=1e12 
mean = 0
var = maxDistance**4/np.log(N)**4 * np.pi**2 / 24
times = np.array(times) - np.mean(times)
tvals = np.linspace(min(times), max(times), num=5000)

best_fit = gumbelPDF(tvals, np.mean(times), np.var(times))
our_fit = gumbelPDF(tvals, mean, var)
t, theory = getNParticlePDF(maxDistance, N)
t_adjust = t - np.sum(t * theory)
print(np.sum(theory))

t_firstOrder, theoryFirstOrder = getNParticlePDF(maxDistance, N)
t_adjustFirstOrder = t_firstOrder - np.sum(t_firstOrder * theoryFirstOrder)

fig, ax = plt.subplots()
ax.hist(times, bins=100, density=True)
ax.plot(tvals, best_fit, ls='--', c='r', label='Best Fit Gumbel Distribution')
ax.plot(tvals, our_fit, ls='--', label='My Derived Gumbel Distribution')
ax.plot(t_adjust[1:], theory[1:] / np.diff(t), ls='--', label='SSRW Thoeretical Curve')
ax.plot(t_adjustFirstOrder[1:], theoryFirstOrder[1:] / np.diff(t_firstOrder), ls='--', label='SSRW First Order')
ax.set_title(f"L={maxDistance}, N=1e12")
ax.set_ylabel("Probability Density")
ax.set_yscale("log")
ax.set_xlabel(r"$\tau_{\mathrm{Min}} - \mathrm{Mean}(\tau_{\mathrm{Min}})$")
ax.set_xlim([-3e6, 1e6])
ax.set_ylim([10**-10, 3 * 10**-6])
ax.legend()
fig.savefig("HistTime.png", bbox_inches='tight')