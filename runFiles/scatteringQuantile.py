from pyDiffusion.pyscattering import evolveAndGetQuantile
import numpy as np
import sys
import os 
import pandas as pd

if __name__ == '__main__':
    (save_dir, sysID, beta) = sys.argv[1:]
    beta = float(beta)
    save_file = os.path.join(save_dir, f'Quantile{sysID}.txt')
    times = np.arange(0, 10000).astype(int)
    N = 1e5

    quantiles = evolveAndGetQuantile(times, N, 4 * max(times), beta)
    data = np.vstack((times[1:], quantiles)).T
    df = pd.DataFrame(data, columns=['Time', 'Quantile'])
    df.to_csv(save_file, index=False)