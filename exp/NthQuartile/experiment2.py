import sys
sys.path.append('../src')
from pydiffusion import Diffusion
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
    N = 1e25
    num_of_steps = round(3 * np.log(N) ** (5/2))
    d = Diffusion(N, beta=beta, occupancySize=num_of_steps+1, smallCutoff=0, largeCutoff=0)
    save_times = np.geomspace(1, num_of_steps, 1000, dtype=np.int64)
    save_times = np.unique(save_times)
    quartiles = [1/1e2, 1/1e5, 1/1e10, 1/1e15, 1/1e25]
    d.evolveAndSaveQuartile(save_times, quartiles, save_file)

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
