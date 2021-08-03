import sys
import os

sys.path.append(os.path.abspath("../../src"))
sys.path.append(os.path.abspath("../../cDiffusion"))
from pydiffusion import Diffusion
import numpy as np
import npquad


def runExperiment(beta, save_file):
    """
    Run one Diffusion experiment for values of N & beta and then store the edges
    in filename.

    Parameters
    ----------
    beta : float
        Value of beta for beta distribution

    filename : str
        Where to save the edges to.
    """
    N = np.quad("1e4500")
    num_of_steps = 10000000  # just want the time it turns on so t~ln(N)
    d = Diffusion(N, beta=beta, occupancySize=num_of_steps, probDistFlag=True)
    save_times = np.geomspace(1, num_of_steps, 2000, dtype=np.int64)
    save_times = np.unique(save_times)
    quartiles = [np.quad(f"1e3") + np.quad(f"1e5")] + [
        np.quad(f"1e{i}") for i in range(10, 4500, 10)
    ]
    quartiles = [np.quad("1") / i for i in quartiles]
    d.evolveAndSaveQuartile(save_times, quartiles, save_file)


if __name__ == "__main__":
    topDir = sys.argv[1]
    sysID = sys.argv[2]
    save_dir = f"{topDir}/1.0/QuartileTotal/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_file = save_dir + f"Quartiles{sysID}.txt"
    save_file = os.path.abspath(save_file)
    runExperiment(1.0, save_file)
