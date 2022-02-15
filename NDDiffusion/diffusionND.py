from diffusionND import DiffusionND
import numpy as np
import npquad
from matplotlib import pyplot as plt

if __name__ == '__main__':
    d = DiffusionND(1, 5)
    for i in range(5):
        d.iterateTimestep()
        fig, ax = plt.subplots()
        c = ax.imshow(np.array(d.getCDF()).astype(float), cmap='gray')
        fig.colorbar(c)
        fig.savefig(f"{i}.png")
    print(d.getDistance())
    print(d.getTheta())
