from matplotlib import pyplot as plt
import numpy as np
import npquad
import sys
from scipy.special import erf

sys.path.append("../../src")
from pyfirstPassagePDF import FirstPassagePDF
from fileIO import loadArrayQuad
from quadMath import prettifyQuad

beta = np.inf
maxPosition = 250
file = "Data.txt"
N = np.quad("1e24")
logN = np.log(N).astype(float)
times = np.arange(1, 1000000)
pdf = FirstPassagePDF(beta, maxPosition)
pdf.evolveAndSaveFirstPassagePDF(times, file)
data = loadArrayQuad(file)
pdf_distribution = data[:, 1][1::2]
# pdf_distribution = data[:, 1]
times = data[:, 0][1::2].astype(float)

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

print(sum(pdf_distribution))

stack_fit = stackExchange(np.arange(1, 1_000_000), maxPosition, nmax=100)
fig, ax = plt.subplots()
ax.plot(times, pdf_distribution, label='Data')
ax.plot(np.arange(1, 1_000_000), stack_fit, '--', label='Best Fit')
ax.set_xlabel("Time")
ax.set_ylabel("First Passage Probability")
ax.legend()
fig.savefig("PDFFit.png")