from matplotlib import pyplot as plt
from matplotlib import colors
from pyDiffusion.pydiffusionND import DiffusionND

tMax = 510
L = 2
d = DiffusionND(4*[1], tMax, L)
for t in range(tMax-1):
    d.iterateTimestep()
cmap = plt.get_cmap("cool")  # Can be any colormap that you want after the cm
cmap.set_bad(color='white')

cdf = d.CDF.astype(float)
cdf[cdf==0] = -1
fig, ax = plt.subplots()
img = ax.imshow(cdf, cmap=cmap, norm=colors.LogNorm(10**-15, 1))

cbar = fig.colorbar(img, ax=ax)
cbar.ax.set_ylabel("Probability Density")
fig.savefig(f"CDF.png", bbox_inches='tight')
plt.close(fig)