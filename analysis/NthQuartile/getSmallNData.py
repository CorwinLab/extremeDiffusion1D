import sys

sys.path.append("../src")
from cdiffusion import Diffusion
import numpy as np
import os
import time


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
    N = 1e300
    d = Diffusion(N, beta=beta, smallCutoff=0, largeCutoff=0)
    num_of_steps = int(3 * (np.log(float(N)) ** (5 / 2)))
    d.initializeOccupationAndEdges(num_of_steps)
    times = np.geomspace(1, num_of_steps, 5000, dtype=np.int64)
    times = np.unique(times)
    dt = np.diff(times)
    Ns = np.geomspace(1e20, 1e280, 14)
    quartiles = []
    for t in dt:
        d.evolveTimesteps(t, inplace=True)
        quart = [d.getNthquartile(N / i) for i in Ns]
        quartiles.append(quart)
    times = times - 1
    minEdges, maxEdges = d.getEdges()
    maxEdges = np.array(maxEdges)
    maxEdges = maxEdges[times[1:]]
    maxEdges = np.reshape(maxEdges, (len(maxEdges), 1))
    quartiles = np.asarray(quartiles)
    times = times[1:]
    times = np.reshape(times, (len(times), 1))
    return_array = np.hstack((maxEdges, quartiles))
    return_array = np.hstack((times, return_array))
    np.savetxt(save_file, return_array)


if __name__ == "__main__":
    topDir = sys.argv[1]
    sysID = sys.argv[2]
    save_dir = f"{topDir}/1.0/1Large/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_file = save_dir + f"Quartiles{sysID}.txt"
    runExperiment(1.0, save_file)
