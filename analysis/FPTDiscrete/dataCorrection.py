import numpy as np
import os 
import sys 
from matplotlib import pyplot as plt
import glob
import pandas as pd

home_dir = '/home/jacob/Desktop/talapasMount/JacobData/FPTDiscretePaper'
dirs = os.listdir(home_dir)

def calculateMeanVar(files, max_dist, verbose=True):
    average_data = None
    average_data_squared = None
    number_of_files = 0
    pos = None
    for f in files:
        try:
            data = np.loadtxt(f, delimiter=',', skiprows=1) # columns are position, quantile, variance
        except:
            print("Error with:", f)
            continue
        if max(data[:, 0]) < max_dist:
            continue
        '''The N=1e2 data got some differet values for position so this is a quick hack
        if data.shape != (356, 3):
            continue
        '''
        pos = data[:, 0]
        number_of_files += 1
        if average_data is None:
            average_data = data
            average_data_squared = data ** 2
        else:
            average_data += data
            average_data_squared += data ** 2
        if verbose:
            print(f)

    if number_of_files == 0:
        return None
    mean = average_data / number_of_files
    variance = average_data_squared / number_of_files - mean ** 2
    new_df = pd.DataFrame(np.array([pos, mean[:, 1], variance[:, 1]]).T, columns=['Distance', 'Mean', 'Variance'])
    return new_df

max_distances = [2301, 4605, 10642, 18529, 21142]
dirs = [int(dir) for dir in dirs]
for dir, max_dist in zip(dirs, max_distances):
    files = glob.glob(os.path.join(home_dir, str(dir), 'Q*.txt'))
    for f in files:
        try: 
            np.loadtxt(f, skiprows=1, delimiter=',')
        except ValueError:
            df = pd.read_csv(f)
            idx = df[df['Distance'] == 'Distance'].index
            if len(idx) > 1:
                df_prev = df.loc[:idx[0]-1]
                df_prev['Distance'] = df_prev['Distance'].astype(int)
                df_prev.to_csv(f, sep=',', index=False)
                continue
            else: 
                idx = idx[0]
            df_prev = df.loc[:idx-1]
            df_after = df.loc[idx+1:]

            df_prev['Distance'] = df_prev['Distance'].astype(int)
            df_prev['Time'] = df_prev['Time'].astype(int)

            df_after['Distance'] = df_after['Distance'].astype(int)
            df_after['Distance'] = df_after['Distance'].astype(int)
            
            if max(df_prev['Distance']) > max(df_after['Distance']):
                df_prev.to_csv(f, sep=',', index=False)
            else:
                df_after.to_csv(f, sep=',', index=False)
            print("Fixed: ", f)
            