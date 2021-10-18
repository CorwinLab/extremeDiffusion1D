from matplotlib import pyplot as plt
import numpy as np
import npquad
import sys
import glob

sys.path.append("../../src")
from databases import QuartileDatabase
from theory import theoreticalNthQuartVar, NthQuartVarStr, NthQuartVarStrLargeTimes
from quadMath import prettifyQuad

PDF_files = glob.glob("/home/jacob/Desktop/corwinLabMount/CleanData/QSweep/Q*.txt")
PDFdb = QuartileDatabase(PDF_files)

Discrete_files = glob.glob(
    "/home/jacob/Desktop/corwinLabMount/CleanData/MaxPart/Q*.txt"
)
Discretedb = QuartileDatabase(Discrete_files, nParticles=np.quad("1e10"))

run_again = False
if run_again:
    Discretedb.calculateMeanVar(verbose=True)
    np.savetxt("maxMean.txt", Discretedb.maxMean)
    np.savetxt("maxVar.txt", Discretedb.maxVar)
else:
    Discretedb.maxMean = np.loadtxt("maxMean.txt")
    Discretedb.maxVar = np.loadtxt("maxVar.txt")

PDFdb.loadVar("../QSweep/Var.txt")
PDFdb.loadMean("../QSweep/Mean.txt")

# Discretedb.loadVar("../MaxPart/Var.txt")
# Discretedb.loadMean("../MaxPart/Mean.txt")

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Time / log(N)")
ax.set_ylabel("Variance / log(N)^(2/3)")

fig2, ax2 = plt.subplots(ncols=3, sharey=True, figsize=(12,3))
for elem in ax2:
    elem.set_xscale("log")
    elem.set_yscale("log")
    elem.set_xlabel("Time / log(N)")
    elem.set_ylabel("Variance / log(N)^(2/3)")
    elem.grid(True)

ax2[0].set_title("Probability Density")
ax2[1].set_title("Discrete Systems")
ax2[2].set_title("Both PDF and Discrete")

# Plot the PDF graphs
cm = plt.get_cmap("gist_heat")
colors = [cm(1.0 * i / len(PDFdb.quantiles) / 1.5) for i in range(len(PDFdb.quantiles))]
print("PDF quantile range:", ["{:0e}".format(int(i)) for i in PDFdb.quantiles])
for i, quantile in enumerate(PDFdb.quantiles):
    if i == 0:
        label = "PDF Variance"
    else:
        label = None
    logN = np.log(quantile).astype(float)
    ax.plot(
        PDFdb.time / logN, PDFdb.var[:, i] / logN ** (2 / 3), label=label, c=colors[i]
    )
    ax2[0].plot(
        PDFdb.time / logN, PDFdb.var[:, i] / logN ** (2 / 3), label=label, c=colors[i]
    )
    ax2[2].plot(
        PDFdb.time / logN, PDFdb.var[:, i] / logN ** (2 / 3), label=label, c=colors[i]
    )

# Plot the theoretical quantile variance
theory = theoreticalNthQuartVar(quantile, PDFdb.time)
ax.plot(PDFdb.time / logN, theory / logN ** (2 / 3), '--', label=NthQuartVarStr)
ax.plot(
    PDFdb.time / logN,
    (PDFdb.time * np.pi) ** (1 / 2) / 2 / logN ** (2 / 3),
    '--',
    label=NthQuartVarStrLargeTimes,
)
ax.plot(PDFdb.time / logN, PDFdb.time / logN / logN ** (2 / 3), '--', label="Linear " + prettifyQuad(quantile))

ax2[0].plot(PDFdb.time / logN, theory / logN ** (2 / 3), '--', label=NthQuartVarStr)
ax2[0].plot(
    PDFdb.time / logN,
    (PDFdb.time * np.pi) ** (1 / 2) / 2 / logN ** (2 / 3),
    '--',
    label=NthQuartVarStrLargeTimes,
)
ax2[2].plot(PDFdb.time / logN, theory / logN ** (2 / 3), '--', label=NthQuartVarStr)
ax2[2].plot(
    PDFdb.time / logN,
    (PDFdb.time * np.pi) ** (1 / 2) / 2 / logN ** (2 / 3),
    '--',
    label=NthQuartVarStrLargeTimes,
)

# Plot the discrete variance
print("Discrete Number of Particles", "{:e}".format(int(Discretedb.nParticles)))
logN = np.log(Discretedb.nParticles).astype(float)
ax.plot(
    Discretedb.time / logN,
    Discretedb.maxVar / logN ** (2 / 3),
    label="Discrete" + prettifyQuad(Discretedb.nParticles),
    c="purple",
)

ax2[1].plot(
    Discretedb.time / logN,
    Discretedb.maxVar / logN ** (2 / 3),
    label="Discrete" + prettifyQuad(Discretedb.nParticles),
    c="purple",
)
ax2[2].plot(
    Discretedb.time / logN,
    Discretedb.maxVar / logN ** (2 / 3),
    label="Discrete" + prettifyQuad(Discretedb.nParticles),
    c="purple",
)


# Get 1e100 Number of Particles
maxVar = np.loadtxt("../MaxPart100/MaxVar.txt")
times = np.loadtxt("../MaxPart100/Times.txt")
nParticles = np.quad("1e100")
print("Discrete Number of Particles: 1e100")
logN = np.log(nParticles).astype(float)
ax.plot(
    times / logN,
    maxVar / logN **(2/3),
    label='Discrete 1e100',
    c='g',
    alpha=0.8,
)

ax2[1].plot(
    times / logN,
    maxVar / logN **(2/3),
    label='Discrete 1e100',
    c='g',
    alpha=0.8,
)
ax2[2].plot(
    times / logN,
    maxVar / logN **(2/3),
    label='Discrete 1e100',
    c='g',
    alpha=0.8,
)

# Get 1e300 number of particles
maxVar = np.loadtxt("../MaxPart300/MaxVar.txt")
times = np.loadtxt("../MaxPart300/Times.txt")
nParticles = np.quad("1e300")
print("Discrete Number of Particles: 1e300")
logN = np.log(nParticles).astype(float)
ax.plot(
    times / logN,
    maxVar / logN **(2/3),
    label='Discrete 1e300',
    c='y',
    alpha=0.8
)
ax2[1].plot(
    times / logN,
    maxVar / logN **(2/3),
    label='Discrete 1e300',
    c='y',
    alpha=0.8
)
ax2[2].plot(
    times / logN,
    maxVar / logN **(2/3),
    label='Discrete 1e300',
    c='y',
    alpha=0.8
)

# Get theoretical quantile from CDF
nParticles = np.quad("1e100")
logN = np.log(nParticles).astype(float)
DiscreteVar = np.loadtxt("../CDFVar100/DiscreteVariance.txt")
QuantileVar = np.loadtxt("../CDFVar100/QuantileVar.txt")
time = np.loadtxt("../CDFVar100/Times.txt")
ax.plot(
    time / logN,
    (DiscreteVar + QuantileVar) / logN ** (2/3),
    alpha=0.8,
    c='b',
    label='CDF Discrete 1e100'
)
ax2[1].plot(
    time / logN,
    (DiscreteVar + QuantileVar) / logN ** (2/3),
    alpha=0.8,
    c='b',
    label='CDF Discrete 1e100'
)
ax2[2].plot(
    time / logN,
    (DiscreteVar + QuantileVar) / logN ** (2/3),
    alpha=0.8,
    c='b',
    label='CDF Discrete 1e100'
)

ax.set_ylim([10**-3, 10**4])
ax.legend()
ax.grid(True)
print("Done!")
fig.savefig("./figures/Variance.png")
ax2[0].set_ylim([10**-3, 10**4])
ax2[0].legend(fontsize=8)
ax2[1].legend(fontsize=8)
fig2.savefig("./figures/VarianceSplit.png", bbox_inches='tight')
