import sys
import os
sys.path.append(os.path.abspath('../../src'))
sys.path.append(os.path.abspath('../../cDiffusion'))
from pydiffusion import Diffusion
import numpy as np

def runExperiment(beta, save_file):
    '''
    Run one Diffusion experiment for values of N & beta and then store the edges
    in filename.

    Parameters
    ----------
    beta : float
        Value of beta for beta distribution

    filename : str
        Where to save the edges to.
    '''
    N = 1e300
    num_of_steps = round(3 * np.log(N) ** (5/2))
    d = Diffusion(N, beta=beta, occupancySize=num_of_steps, smallCutoff=0, largeCutoff=0)
    save_times = np.geomspace(1, num_of_steps, 1000, dtype=np.int64)
    save_times = np.unique(save_times)
    quartiles = [10 ** i for i in range(20, 280, 20)]
    quartiles = [1/i for i in quartiles]
    d.evolveAndSaveQuartile(save_times, quartiles, save_file)

if __name__ == '__main__':
    topDir = sys.argv[1]
    sysID = sys.argv[2]
    save_dir = f'{topDir}/1.0/1Large/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_file = save_dir + f'Quartiles{sysID}.txt'
    save_file = os.path.abspath(save_file)
    runExperiment(1.0, save_file)
