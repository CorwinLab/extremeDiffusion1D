from math import erf
import numpy as np
import npquad
import sys 
sys.path.append("../../src")
from pyfirstPassagePDF import FirstPassagePDF
from matplotlib import pyplot as plt
from scipy.special import erfinv

maxTime = 100000
maxPosition = 500
beta = np.inf
pdf = FirstPassagePDF(beta, maxPosition)

passageProbability = np.zeros(maxTime)
for i in range(maxTime):
    pdf.iterateTimeStep()
    pdf_arr = pdf.getPDF()
    passageProbability[i]= pdf.firstPassageProbability

def levyDistribution(t, position, D=1/2):
    return position / np.sqrt(4*np.pi*D*t**3) * np.exp(-position**2/(4*D*t)) 

def longTimePDF(t, position):
    l = np.pi**2 / 8 / position**2
    return l * np.exp(-l * t)

fig, ax = plt.subplots()
time = np.array(range(maxTime))[1::2]
print(np.nansum(levyDistribution(time, maxPosition)))
ax.plot(time, passageProbability[1::2], label='SSRW')
ax.plot(time[1:], levyDistribution(time[1:], maxPosition), label='Levy Distribtuion')
ax.plot(time[1:], longTimePDF(time[1:], maxPosition), label='Redner Theory (2.4.22)')
print(np.sum(passageProbability[1::2]))
print(np.sum(levyDistribution(time, maxPosition)))
print(np.sum(longTimePDF(time, maxPosition)))
ax.set_yscale("log")
ax.set_ylim([10**-20, 10**-2])
ax.set_xlabel("Time")
ax.set_ylabel(r"$\log_2(\mathrm{First Passage Probability})$")
ax.legend()
fig.savefig("FirstPassageProbability.png", bbox_inches='tight')