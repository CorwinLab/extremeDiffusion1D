import sys
sys.path.append('../src')
from cdiffusion import Diffusion
import numpy as np
from matplotlib import pyplot as plt
import os
import time

def runExperiment(N, filename):
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

    d = Diffusion(numberOfParticles=N, beta=1.0)
    num_of_steps = round(np.log(float(N)) ** (5/2))
    d.evolveEinstein(int(num_of_steps))
    edges = np.array(d.getEdges()).T
    np.savetxt(filename, edges)

if __name__ == '__main__':
    topDir = sys.argv[1]
    numWalkersStr = sys.argv[2]
    numWalkers = int(float( numWalkersStr ))
    sysID = sys.argv[3]
    numWalkers_string = "{:.2e}".format(numWalkers).replace("+", "_")
    check_dir = f'{topDir}/Einstein/' + numWalkers_string
    if not os.path.exists(check_dir):
        os.makedirs(check_dir)
    save_file = f'{topDir}/Einstein/' + numWalkers_string + f'/Edges{sysID}.txt'
    start = time.time()
    runExperiment(numWalkers, save_file)
