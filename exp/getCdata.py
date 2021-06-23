import sys
sys.path.append('../src')
from cdiffusion import Diffusion
import numpy as np
from matplotlib import pyplot as plt
import os

def runExperiment(N, beta, filename):
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

    d = Diffusion(numberOfParticles=N, beta=beta)
    d.initializeOccupation()
    num_of_steps = round(np.log(N) ** (5/2))
    d.evolveTimesteps(num_of_steps)
    edges = np.array(d.getEdges()).T
    np.savetxt(filename, edges)

if __name__ == '__main__':
    topDir = sys.argv[1]
    numWalkersStr = sys.argv[2]
    numWalkers = int( numWalkersStr )
    beta = float(sys.argv[3])
    sysID = sys.argv[4]
    save_file = f'{topDir}/{beta}/{numWalkers}/Edges{sysID}.txt'
    runExperiment(numWalkers, beta, save_file)
