import sys
import os

sys.path.append(os.path.abspath("../../src"))
sys.path.append(os.path.abspath("../../cDiffusion"))
from pydiffusion import Diffusion
import numpy as np


def runExperiment(beta, save_file):
    """
    Run one Diffusion experiment for values of N & beta and then store the edges
    in filename.

    Parameters
    ----------
    N : integer
        Number of particles in experiment

    beta : float
        Value of beta for beta distribution

    filename : str
        Where to save the edges to.
    """
    N = 1
    num_of_steps = 10_000
    d = Diffusion(
        N,
        beta=beta,
        occupancySize=num_of_steps,
        smallCutoff=0,
        largeCutoff=0,
        probDistFlag=False,
    )
    d.evolveTimeSteps(num_of_steps)
    maxDist = d.maxDistance
    time = d.center * 2
    return_array = np.vstack((time, maxDist)).T
    np.savetxt(save_file, return_array)


if __name__ == "__main__":
    for i in range(2500):
        save_file = f"./Data/Data{i}.txt"
        save_file = os.path.abspath(save_file)
        runExperiment(0.0, save_file)
        print("Finished: ", i)
