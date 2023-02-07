from pyDiffusion.pydiffusionND import DiffusionND
import numpy as np
import npquad
from matplotlib import pyplot as plt 

tMax = 10
d = DiffusionND([1, 1, 1, 1], tMax, 100)
#print(d.CDF)
for _ in range(tMax-1):
    d.iterateTimestep()
    print(d.CDF)

fig, ax = plt.subplots()
ax.imshow(np.array(d.getCDF()).astype(float))
fig.savefig("CDF.png")

print("Random number sum:", sum(d.getRandomNumbers()))