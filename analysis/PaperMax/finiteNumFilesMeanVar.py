import numpy as np
import npquad
import pandas as pd
import sys 
import os
import glob

def calculateMeanVarHelper(
    files, skiprows=1, delimiter=",", verbose=False, maxTime=None, nFiles=None,
):
    """
    Calculate mean and variance of arrays in files.
    """
    squared_sum = None
    sum = None
    return_time = None
    number_of_files = 0
    for f in files:
        try:
            data = np.loadtxt(f, delimiter=delimiter, skiprows=skiprows)
        except StopIteration:
            continue
        # data = fileIO.loadArrayQuad(f, delimiter=delimiter, skiprows=skiprows)
        df = pd.DataFrame(data.astype(float))
        df = df.drop_duplicates(subset=[0], keep="last")

        data = df.to_numpy()
        time = data[:, 0].astype(np.float64)
        data = data[:, 1:]

        if maxTime is not None:
            if max(time) < maxTime:
                continue
            elif max(time) == maxTime:
                time = time[time <= maxTime]
                if return_time is not None:
                    if len(time) < len(return_time):
                        print("Missing times:", np.setdiff1d(return_time, time))
                        continue
                return_time = time
        else:
            maxTime = max(time)

        maxIdx = len(time)
        data = data[:maxIdx, :]

        if squared_sum is None:
            squared_sum = np.zeros(data.shape, dtype=np.quad)
        if sum is None:
            sum = np.zeros(data.shape, dtype=np.quad)

        squared_sum += data ** 2
        sum += data
        number_of_files += 1
        if nFiles is not None:
            if number_of_files == nFiles:
                break

        if verbose:
            print(f, time.shape)

    if return_time is None:
        return_time = time

    mean = (sum / number_of_files).astype(np.float64)
    var = (squared_sum / number_of_files).astype(np.float64) - mean ** 2

    return return_time.ravel(), mean.ravel(), var.ravel(), maxTime, number_of_files

max_dir = "/home/jacob/Desktop/corwinLabMount/CleanData/Paper/Max/"
dirs = os.listdir(max_dir)
for dir in dirs:
    path = os.path.join(max_dir, dir, 'Q*.txt')
    files = glob.glob(path)
    time, mean, var, maxTime, nFiles = calculateMeanVarHelper(files, verbose=False, nFiles=1000)
    df = pd.DataFrame(np.array([time, mean, var]).T, columns=['time', 'Mean Max', 'Var Max'])
    print(dir, ':', nFiles)
    df.to_csv(os.path.join(max_dir, dir, f'MeanVar{nFiles}.txt'), index=False)
