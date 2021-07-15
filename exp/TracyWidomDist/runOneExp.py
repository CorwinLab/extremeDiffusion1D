import sys
import os
sys.path.append(os.path.abspath('../../src'))
sys.path.append(os.path.abspath('../../cDiffusion'))
from pydiffusion import Diffusion
import numpy as np
import pandas as pd

def runExperiment(beta, save_file):
    N = 1
    num_of_steps = 1_000
    d = Diffusion(N, beta=beta, occupancySize=num_of_steps, smallCutoff=0, largeCutoff=0, probDistFlag=True)
    save_times = np.geomspace(1, num_of_steps, 100, dtype=np.int64)
    save_times = np.unique(save_times)
    vs = np.arange(0, 1.1, 0.1)
    d.evolveAndSaveV(save_times, vs, save_file)

if __name__ == '__main__':
    runExperiment(1.0, 'Data.txt')
    data = pd.read_csv('Data.txt')
    print(data.tail())
