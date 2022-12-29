import numpy as np
from pyDiffusion import pydiffusion2D
from matplotlib import pyplot as plt
import pandas as pd

if __name__ == '__main__':
    nParticles = 10000
    save_times = np.arange(5, 1000, 5)
    xi = 3 
    save_file = "MaxPosition.txt"
    save_positions = "FinalPositions.txt"
    pydiffusion2D.evolveAndSaveMaxDistance1D(nParticles, save_times, xi, save_file, save_positions)
    max_positions = np.loadtxt("FinalPositions.txt")
    data = pd.read_csv("MaxPosition.txt")

    fig, ax = plt.subplots()
    ax.hist(max_positions, bins=50, density=True)
    ax.set_xlabel("Position")
    ax.set_ylabel("Probability Density")
    ax.set_yscale("log")
    fig.savefig("Positions.png", bbox_inches='tight')

    fig, ax = plt.subplots()
    ax.plot(data['Time'], data['Position'])
    ax.set_xlabel("Time")
    ax.set_ylabel("Position")
    fig.savefig("MaxParticle.png", bbox_inches='tight')
