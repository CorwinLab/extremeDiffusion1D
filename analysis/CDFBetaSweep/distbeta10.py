import numpy as np 
import sys 
from matplotlib import pyplot as plt
sys.path.append("../../src")
from libDiffusion import RandomDistribution

# Get Beta=100 distributions
beta = RandomDistribution('beta', [10, 10])

var = 1/84
n=3
a = 1/2*(1-np.sqrt(12*n*var))
b = 1-a
bates = RandomDistribution('bates', [n, a, b])

a = 1/2 * (1 - 1/np.sqrt(7))
b = 1-a
uniform = RandomDistribution('uniform', [a, b])

a = 1/2 * (1 - np.sqrt(5/63))
b = 1-a
quadratic = RandomDistribution('quadratic', [a, b])

A = 1/42
delta = RandomDistribution('delta', [A, 1-2*A, A])

num_samples = 10_000_000
beta_vals = np.zeros(num_samples)
bates_vals = np.zeros(num_samples)
uniform_vals = np.zeros(num_samples)
quadratic_vals = np.zeros(num_samples)
delta_vals = np.zeros(num_samples)

for i in range(num_samples):
    beta_vals[i] = beta.generateRandomVariable()
    bates_vals[i] = bates.generateRandomVariable()
    uniform_vals[i] = uniform.generateRandomVariable()
    quadratic_vals[i] = quadratic.generateRandomVariable()
    delta_vals[i] = delta.generateRandomVariable()

# Check all values and distributions are reasonable
bins = np.linspace(0, 1, 1000)
fig, ax = plt.subplots()
ax.hist(delta_vals, label='Delta', bins=bins, density=True, histtype='step', lw=2)
ax.hist(beta_vals, label='Beta', bins=bins, density=True, histtype='step')
ax.hist(bates_vals, label='Bates', bins=bins, density=True, histtype='step')
ax.hist(uniform_vals, label='Uniform', bins=bins, density=True, histtype='step')
ax.hist(quadratic_vals, label='Quadratic', bins=bins, density=True, histtype='step')
ax.set_xlim([0, 1])
ax.set_ylim([10**-2, 2*10**3])
ax.set_yscale("log")
ax.set_ylabel("Prbability Density")
ax.set_xlabel(r"$x$")
ax.set_title(r"$\sigma^2_w = \frac{1}{84}$")
ax.legend()
fig.savefig("Beta10.pdf", bbox_inches='tight')

print('Delta:', np.var(delta_vals))
print('Beta:', np.var(beta_vals))
print('Bates:', np.var(bates_vals))
print('Uniform:', np.var(uniform_vals))
print('Quadratic:', np.var(quadratic_vals))
print('Real Distribution:', 1/84)