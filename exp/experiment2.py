import sys
sys.path.append('../src')
from cdiffusion import Diffusion
import numpy as np
import os
import time

def runExperiment(beta, save_file):
    '''
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
    '''
    N = 1e10
    d = Diffusion(N, beta=beta, smallCutoff=0, largeCutoff=0)
    num_of_steps = int( 3 * (np.log(float(N)) ** (5/2)) )
    d.initializeOccupationAndEdges(num_of_steps)
    times = np.geomspace(1, num_of_steps, 5000, dtype=np.int64)
    times = np.unique(times)
    Ns = np.geomspace(1e20, 1e280, 14)
    quartiles = []
    count = 0
    elapsed_time = 0
    prev_idx = 0
    for j, t in enumerate(times):
        d.evolveToTime(t, inplace=True)
        quart = [d.getNthquartile(N / i) for i in Ns]
        quartiles.append(quart)

        # Save quartiles every couple of steps in time
        elapsed_time += t
        if (elapsed_time // 100) > count:
            count = (elapsed_time // 100)

            _, maxEdges = d.getEdges()
            append_times = times[prev_idx : j + 1] - 1
            maxEdges = np.asarray(maxEdges)
            maxEdges = maxEdges[append_times]

            maxEdges = np.reshape(maxEdges, (len(maxEdges), 1))
            append_times = np.reshape(append_times, (len(append_times), 1))
            quartiles = np.asarray(quartiles)

            return_array = np.hstack((maxEdges, quartiles))
            return_array = np.hstack((append_times, return_array))

            f = open(save_file, 'a')
            np.savetxt(f, return_array)
            f.close()

            quartiles = []
            prev_idx = j + 1

    data = np.loadtxt(save_file)
    t = data[:, 0]
    assert np.all(t == (times-1))

if __name__ == '__main__':
    '''
    topDir = sys.argv[1]
    sysID = sys.argv[2]
    save_dir = f'{topDir}/1.0/1Large/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_file = save_dir + f'Quartiles{sysID}.txt'
    '''
    runExperiment(1.0, 'Data.txt')
