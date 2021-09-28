from matplotlib import pyplot as plt
import numpy as np
import npquad
import sys
import glob

sys.path.append("../../src")
from databases import QuartileDatabase
from theory import theoreticalNthQuartVar, NthQuartVarStr, NthQuartVarStrLargeTimes

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

# Plot the PDF graphs
cm = plt.get_cmap("gist_heat")
colors = [cm(1.0 * i / len(PDFdb.quantiles) / 1.5) for i in range(len(PDFdb.quantiles))]
for i, quantile in enumerate(PDFdb.quantiles):
    if i == 0:
        label = "PDF Variance"
    else:
        label = None
    logN = np.log(quantile).astype(float)
    ax.plot(
        PDFdb.time / logN, PDFdb.var[:, i] / logN ** (2 / 3), label=label, c=colors[i]
    )

# Plot the theoretical quantile variance
theory = theoreticalNthQuartVar(quantile, PDFdb.time)
ax.plot(PDFdb.time / logN, theory / logN ** (2 / 3), label=NthQuartVarStr)
ax.plot(
    PDFdb.time / logN,
    (PDFdb.time * np.pi) ** (1 / 2) / 2 / logN ** (2 / 3),
    label=NthQuartVarStrLargeTimes,
)

# Plot the discrete variance
logN = np.log(Discretedb.nParticles).astype(float)
ax.plot(
    Discretedb.time / logN,
    Discretedb.maxVar / logN ** (2 / 3),
    label="Discrete",
    c="purple",
)
ax.plot(PDFdb.time / logN, PDFdb.time / logN / logN ** (2 / 3), label="Linear")

gumbel_quantiles = PDFdb.quantiles[1:-1]
cm = plt.get_cmap("winter")
colors = [
    cm(1.0 * i / len(gumbel_quantiles) / 1.5) for i in range(len(gumbel_quantiles))
]
# Plot the Gumbel Results
for i, N in enumerate(gumbel_quantiles):
    if i == 0:
        label = r"$(10N - 0.1N)^{2}$"
    else:
        label = None
    logN = np.log(N).astype(float)
    diff = PDFdb.getGumbalDiff(N) ** 2
    ax.plot(PDFdb.time / logN, diff / logN ** (2 / 3), c=colors[i], label=label)

times = np.loadtxt("../CDFVar/Times.txt")
discreteVar = np.loadtxt("../CDFVar/DiscreteVariance.txt")
quantileVar = np.loadtxt("../CDFVar/QuantileVar.txt")
N = 1000000

ax.plot(times / np.log(N), (discreteVar+quantileVar) / (np.log(N)**(2/3)), label='Discrete from CDF')

ax.set_ylim([10**-3, 10**4])
ax.legend()
ax.grid(True)

fig.savefig("./figures/Variance.png")
