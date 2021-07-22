import sys
import os
sys.path.append(os.path.abspath('../../src'))
sys.path.append(os.path.abspath('../../cDiffusion'))
from pydiffusion import Diffusion
import numpy as np

def runExperiment(save_file):
    N = 100
    num_of_steps = int(np.log(N) ** (5/2))
    d = Diffusion(N, beta=np.inf, occupancySize=num_of_steps, probDistFlag=False)
    for i in range(num_of_steps):
        d.iterateTimestep()
    _, maxEdge = d.getEdges()
    np.savetxt(save_file, maxEdge)

if __name__ == '__main__':
    #topDir = sys.argv[1]
    #sysId = sys.argv[2]
    topDir = '.'
    sysId = '1'
    save_dir = os.path.join(topDir, 'EinsteinData')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_file = os.path.join(save_dir, f'Einstein{sysId}.txt')
    runExperiment(save_file)
