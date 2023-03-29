from pyDiffusion.pyscattering import evolveAndGetQuantile
import numpy as np
import sys
import os 
import pandas as pd

if __name__ == '__main__':
    (save_dir, sysID) = sys.argv[1:]
    save_file = os.path.join(save_dir, f'Quantile{sysID}.txt')
    times = np.arange(0, 1000).astype(int)
    N = 1e10 

    quantiles = evolveAndGetQuantile(times, N, 4 * max(times))
    df = pd.DataFrame(np.array([times, quantiles]).T, columns=['Time', 'Quantile'])
    df.to_csv(save_file, index=False)