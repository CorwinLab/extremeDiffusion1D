from matplotlib import pyplot as plt
import numpy as np
import npquad
import sys
from scipy.special import erf

sys.path.append("../../src")
from pyfirstPassagePDF import FirstPassagePDF
from fileIO import loadArrayQuad
from quadMath import prettifyQuad


def stackExchange(t, L, nmax=10):
    sum = 0
    for k in np.arange(-nmax, nmax):
        phi = L * (1 + 2 * k)
        sum += (
            (-1) ** abs(k)
            * phi
            / np.sqrt(2 * np.pi * t ** 3)
            * np.exp(-((phi) ** 2) / 2 / t)
        )
    return sum

def jacob(t, L, nmax=10):
    sum = 0
    for k in np.arange(-nmax, nmax):
        sum += (
            np.sqrt(2)
            * L
            / np.sqrt(np.pi * t ** 3)
            * (-1) ** (abs(k + 1))
            * (k - 1 / 2)
            * np.exp(-2 * L ** 2 / t * (k - 1 / 2) ** 2)
        )
    return sum


def stackExchange2(t, L, nmax=10):
    sum = 0
    for n in range(nmax):
        sum += (
            n
            * np.pi
            / L ** 2
            * (1 - (-1) ** n)
            * np.sin(n * np.pi / 2)
            * np.exp(-(n ** 2) * np.pi ** 2 * t / (2 * L ** 2))
        )
    return sum


def continousPDF(t, L, nmax=10, D=1 / 2):
    sum = 0
    for n in range(0, nmax):
        sum += (
            (-1) ** (n + 1)
            * L ** 2
            / ((2 * n + 1) * np.pi * D)
            * np.exp(-((n + 1 / 2) ** 2) * np.pi ** 2 * D * t / L ** 2)
        )
    return sum / np.sum(sum)

def jacobCDF(t, L, nmax=100): 
    sum = 0 
    for k in range(-nmax, nmax): 
        sum += - (-1)**(abs(k)) * (k+1/2)/abs(k+1/2) * erf(np.sqrt(2*L**2 *(k+1/2)**2 / t))
    return sum

def jacobCDFExp(t, L, nmax=100): 
    sum = 0 
    for k in range(-nmax, nmax): 
        x = np.sqrt(2*L**2 *(k+1/2)**2 / t)
        sum += - (-1)**(abs(k)) * (k+1/2)/abs(k+1/2) / np.sqrt(np.pi) * np.exp(-x**2)/x
    return sum

beta = np.inf
maxPosition = 250
file = "Data.txt"
N = np.quad("1e24")
logN = np.log(N).astype(float)
times = np.arange(1, 100_000)
pdf = FirstPassagePDF(beta, maxPosition)
pdf.evolveAndSaveFirstPassagePDF(times, file)
data = loadArrayQuad(file)
pdf_distribution = data[:, 1][1::2]
# pdf_distribution = data[:, 1]
times = data[:, 0][1::2].astype(float)
# times = data[:, 0].astype(float)
cdf_distribution = np.cumsum(pdf_distribution)
N_particle_cdf = 1 - np.exp(-cdf_distribution * N)
N_particle_pdf = np.diff(N_particle_cdf)

redner_fit = continousPDF(times, maxPosition, nmax=1000)
stack_fit = stackExchange(times, maxPosition, nmax=1000)
stack_fit2 = stackExchange2(times, maxPosition * 2, nmax=1000)
jacob_fit = jacob(times, maxPosition, nmax=1000)
jacob_cdf_fit = jacobCDF(times, maxPosition, nmax=1000)
jacob_cdf_exp = jacobCDFExp(times, maxPosition)

print(sum(stack_fit))
print(sum(pdf_distribution))


def calculateMeanAndVariance(x, pdf):
    mean = sum(x * pdf)
    var = sum(x ** 2 * pdf) - mean ** 2
    return mean, var


mean, var = calculateMeanAndVariance(times, pdf_distribution)
meanN, varN = calculateMeanAndVariance(times[1:], N_particle_pdf)

fig, ax = plt.subplots(ncols=2, nrows=2, sharex=True)
fig.suptitle(f"First Passage Probabilities for Distance={maxPosition}", fontsize=16)
ax[0][0].plot(times, pdf_distribution.astype(float))
ax[0][0].plot(times, stack_fit.astype(float))
ax[0][0].plot(times, stack_fit2.astype(float))
ax[0][0].plot(
    np.arange(1, 500_000), stackExchange(np.arange(1, 500_000), maxPosition, nmax=1000)
)
ax[0][0].set_ylim([10 ** -8, 10 ** -4])
# ax[0][0].plot([mean.astype(float), mean.astype(float)], [0, max(pdf_distribution.astype(float))], color='r')
ax[0][0].set_ylabel("PDF")
ax[0][0].set_title("Single Particle")
ax[1][0].plot(times, cdf_distribution)
ax[1][0].plot(times, jacob_cdf_fit, '--')
ax[1][0].plot(times, jacob_cdf_exp, '--')
ax[1][0].set_ylabel("CDF")
ax[1][0].set_xlabel("Time")
ax[0][1].set_title(f"N={prettifyQuad(N)}")
ax[1][1].plot(times, N_particle_cdf)
ax[1][1].set_xscale("log")
ax[0][1].plot(times[1:], N_particle_pdf)
ax[0][1].plot(
    [meanN.astype(float), meanN.astype(float)],
    [0, max(N_particle_pdf.astype(float))],
    color="r",
)
ax[0][1].set_yscale("log")
plt.tight_layout()
fig.savefig("PDF.png", bbox_inches="tight")

fig, ax = plt.subplots()
ax.set_yscale("log")
ax.set_xlabel("Time")
ax.set_ylabel("Probability Density of First Passage")
ax.plot(times, pdf_distribution.astype(float))
ax.plot(times, 2 * stack_fit.astype(float))
ax.plot(times, 2 * jacob_fit.astype(float), c="m", ls="--")
ax.set_ylim([10 ** -20, 10 ** -4])
fig.savefig("Prob.png")
