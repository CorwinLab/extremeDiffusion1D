import numpy as np 
import sys 
from matplotlib import pyplot as plt
sys.path.append("../../src")
from libDiffusion import RandomDistribution

# Get Beta=100 distributions
beta = RandomDistribution('beta', [1, 1])

a = 1/2 * (1 - np.sqrt(5/9))
b = 1-a
quadratic = RandomDistribution('quadratic', [a, b])

A = 1/6
delta = RandomDistribution('delta', [A, 1-2*A, A])

num_samples = 10_000_000
beta_vals = np.zeros(num_samples)
quadratic_vals = np.zeros(num_samples)
delta_vals = np.zeros(num_samples)

for i in range(num_samples):
    beta_vals[i] = beta.generateRandomVariable()
    quadratic_vals[i] = quadratic.generateRandomVariable()
    delta_vals[i] = delta.generateRandomVariable()

delta_color = 'tab:blue'
beta_color = 'tab:red'
quad_color = 'tab:orange'
fontsize = 14
lw = 2

# Check all values and distributions are reasonable
bins = np.linspace(0, 1, 1000)
fig, ax = plt.subplots()
ax.hist(delta_vals, label='Delta', bins=bins, density=True, histtype='step', lw=lw, color=delta_color)
ax.hist(beta_vals, label='Beta', bins=bins, density=True, histtype='step', lw=lw, color=beta_color)
ax.hist(quadratic_vals, label='Quadratic', bins=bins, density=True, histtype='step', lw=lw, color=quad_color)
ax.set_xlim([0, 1])
ax.set_ylim([10**-2, 2*10**3])
ax.set_yscale("log")
ax.set_ylabel("Prbability Density")
ax.set_xlabel(r"$x$")
ax.set_title(r"$\sigma^2_w = \frac{1}{12}, c_2 = 1$")

leg = ax.legend(
    loc="upper left",
    framealpha=0,
    labelcolor=[delta_color, beta_color, quad_color],
    handlelength=0,
    handletextpad=0,
    fontsize=fontsize,
)
for item in leg.legendHandles:
    item.set_visible(False)
    
fig.savefig("Beta1.pdf", bbox_inches='tight')

print('Delta:', np.var(delta_vals))
print('Beta:', np.var(beta_vals))
print('Quadratic:', np.var(quadratic_vals))
print('Real Distribution:', 1/12)
print('c_2: ', 1/1**2)