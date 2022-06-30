from diffusionCDF import DiffusionTimeCDF
import numpy as np 
import npquad
import matplotlib.pyplot as plt

beta = 100
tMax = 10
num_of_vals = 10_000_000
d = DiffusionTimeCDF(beta, tMax)
vals = np.zeros(shape=num_of_vals)
for i in range(num_of_vals):
    vals[i] = d.generateBeta()

print('Number of NaN vals:', np.count_nonzero(np.isnan(vals)))
print("Mean:", np.nanmean(vals))
fig, ax = plt.subplots()
ax.hist(vals, bins=500, density=True)
ax.set_yscale("log")
ax.set_xlim([0, 1])
fig.savefig("Distribution.png", bbox_inches='tight')