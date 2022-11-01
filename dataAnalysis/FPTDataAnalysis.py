import numpy as np
import pandas as pd

def calculateMeanVarDiscrete(files, max_dist, verbose=True):
    average_data = None
    average_data_squared = None
    number_of_files = 0
    pos = None
    for f in files:
        try:
            data = pd.read_csv(f) # columns are Distance, Time
        except:
            print("Error with:", f)
            continue

        if max(data['Distance']) < max_dist:
            continue

        data = data[data['Distance'] <= max_dist]
        data = data.values
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

    mean = average_data[:, 1] / number_of_files
    variance = average_data_squared[:, 1] / number_of_files - mean ** 2

    forth_moment = None
    forth_moment_files = 0
    for f in files:
        try:
            data = pd.read_csv(f) # columns are Distance, Time
        except:
            print("Error with:", f)
            continue
        if max(data['Distance']) < max_dist:
            continue

        data = data[data['Distance'] <= max_dist]
        assert np.array_equal(data['Distance'].values, pos)
        forth_moment_files += 1
        if forth_moment is None:
            forth_moment = (data['Time'].values - mean) ** 4
        else: 
            forth_moment += (data['Time'].values - mean) ** 4
        if verbose:
            print(f)
            
    forth_moment = (forth_moment / forth_moment_files - variance ** 2) / forth_moment_files
    new_df = pd.DataFrame(np.array([pos, mean, variance, forth_moment]).T, columns=['Distance', 'Mean', 'Variance', 'Forth Moment'])
    return new_df, number_of_files

def calculateMeanVarCDF(files, max_dist, verbose=True):
    average_data = None
    average_data_squared = None
    number_of_files = 0
    pos = None
    for f in files: 
        data = pd.read_csv(f) # columns are Position, Quantile, Variance
        if max(data['Position']) < max_dist:
            print("Not Enough Data: ", f)
            continue
        data = data[data['Position'] <= max_dist]
        data = data.values
        pos = data[:, 0]
        number_of_files += 1
        if average_data is None:
            average_data = data
            average_data_squared = data ** 2
        else:
            average_data += data
            average_data_squared += data ** 2
    if number_of_files == 0:
        return None
    mean = average_data / number_of_files
    variance = average_data_squared / number_of_files - mean ** 2
    new_df = pd.DataFrame(np.array([pos, mean[:, 1], mean[:, 2], variance[:, 1]]).T, columns=['Distance', 'Mean Quantile', 'Sampling Variance', 'Env Variance'])
    return new_df, number_of_files
