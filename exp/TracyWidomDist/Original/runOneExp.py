import sys
import os

sys.path.append(os.path.abspath("../../src"))
sys.path.append(os.path.abspath("../../cDiffusion"))
from pydiffusion import Diffusion
import numpy as np
import npquad


def runExperiment(beta, save_file):
    N = np.quad("1e4500")
    num_of_steps = 160000  # prob seems to zero out here for v~0.5, 0.7
    d = Diffusion(N, beta=beta, occupancySize=num_of_steps, probDistFlag=True)
    save_times = np.geomspace(1, num_of_steps, 5000, dtype=np.int64)
    save_times = np.unique(save_times)
    vs = np.geomspace(1e-7, 1, 50)
    d.evolveAndSaveV(save_times, vs, save_file)


if __name__ == "__main__":
    topDir = sys.argv[1]
    sysId = sys.argv[2]
    save_dir = os.path.join(topDir, "1.0/TracyWidomN8000")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_file = os.path.join(save_dir, f"TracyData{sysId}.txt")
    runExperiment(1.0, save_file)
