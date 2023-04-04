from pyDiffusion.pyscattering import evolveAndGetQuantile
import numpy as np
import sys
import os 
import pandas as pd

if __name__ == '__main__':
    # For testing purposes 
    # save_dir, sysID, beta = '.', '1', '1'
    (save_dir, sysID, beta) = sys.argv[1:]
    beta = float(beta)
    save_file = os.path.join(save_dir, f'Quantile{sysID}.txt')
    times = np.geomspace(1, 1e5, 5000).astype(int)
    times = np.unique(times)
    N = 1e5

    quantiles = evolveAndGetQuantile(times, N, 4 * max(times), beta, save_file)