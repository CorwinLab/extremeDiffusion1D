import numpy as np 
import sys 
from matplotlib import pyplot as plt
sys.path.append("../../src")
from libDiffusion import RandomDistribution

# Get Beta=100 distributions
beta = RandomDistribution('beta', [0.01, 0.01])

A = 25/51
delta = RandomDistribution('delta', [A, 1-2*A, A])

c = 1 - np.sqrt(1-6*(1/4-25/102))
inv_triang = RandomDistribution('inv triangular', [c])

num_samples = 10_000_000
beta_vals = np.zeros(num_samples)
delta_vals = np.zeros(num_samples)
inv_triang_vals = np.zeros(num_samples)

for i in range(num_samples):
    beta_vals[i] = beta.generateRandomVariable()
    delta_vals[i] = delta.generateRandomVariable()
    inv_triang_vals[i] = inv_triang.generateRandomVariable()

# Check all values and distributions are reasonable
bins = np.linspace(0, 1, 1000)
fig, ax = plt.subplots()
ax.hist(delta_vals, label='Delta', bins=bins, density=True, histtype='step', lw=2)
ax.hist(beta_vals, label='Beta', bins=bins, density=True, histtype='step')
ax.hist(inv_triang_vals, label='Inv Triangle', bins=bins, density=True, histtype='step')
ax.set_xlim([0, 1])
ax.set_ylim([10**-2, 2*10**3])
ax.set_yscale("log")
ax.set_ylabel("Prbability Density")
ax.set_xlabel(r"$x$")
ax.set_title(r"$\sigma^2_w = \frac{25}{102}$")
ax.legend()
fig.savefig("Beta001.pdf", bbox_inches='tight')

print('Delta:', np.var(delta_vals))
print('Beta:', np.var(beta_vals))
print('Inv Triangular:', np.var(inv_triang_vals))
print('Real Distribution:', 25/102)