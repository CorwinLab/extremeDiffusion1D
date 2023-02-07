from diffusionND import DiffusionND
import numpy as np
import npquad
from matplotlib import pyplot as plt 

tMax = 10
d = DiffusionND([1, 1, 1, 1], tMax)

for _ in range(tMax):
    d.iterateTimestep()

fig, ax = plt.subplots()
ax.imshow(np.array(d.getCDF()).astype(float))
fig.savefig("CDF.png")

print(sum(d.getRandomNumbers()))