import sys 
sys.path.append("../src")
from libDiffusion import RandomNumGenerator
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import beta as betaFunction

beta=100
r = RandomNumGenerator(beta)
vals = []
for _ in range(10_000_000):
    vals.append(r.generateBeta())
print("Number of NaN values:", np.sum(np.isnan(vals)))
print(np.min(vals), np.max(vals))
x = np.linspace(0, 1, num=100000)
beta_pdf = betaFunction.pdf(x, beta, beta)
fig, ax = plt.subplots()
ax.hist(vals, bins=1000, density=True)
ax.plot(x, beta_pdf, ls='--', alpha=0.5, label=f'Beta({beta})')
ax.set_xlabel("x")
ax.set_ylabel("Probability Density")
ax.set_xlim([0, 1])
ax.set_yscale("log")
ax.set_ylim([10**-5, 20])
ax.legend()
fig.savefig("Beta100Histogram.pdf", bbox_inches='tight')