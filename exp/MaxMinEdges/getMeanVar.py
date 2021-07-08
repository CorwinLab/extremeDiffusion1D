import numpy as np
import glob
import os

topdir = '/home/jhass2/Data'

def get_files(beta, N):
    N_str = "{:.2e}".format(N).replace('+', '_')
    search_dir = os.path.join(topdir, str(beta))
    search_dir = os.path.join(search_dir, N_str)
    file_search = os.path.join(search_dir, 'Edges*.txt')
    files = glob.glob(file_search)
    return files

def get_beta_runs():
    betas = os.listdir(topdir)
    betas = [x for x in betas if x != 'log']
    return betas

def mean_max_from_files(files):
    if not files:
        return
    top_dir = os.path.dirname(files[0])
    mean_save = os.path.join(top_dir, "max_mean.txt")
    var_save = os.path.join(top_dir, "max_variance.txt")
    
    all_data = None

    for file in files:
        data = np.loadtxt(file)
        center = np.arange(1, len(data[:, 0])+1) * 0.5
        min_edge = data[:, 0]
        max_edge = data[:, 1]
        minDist = abs(min_edge - center)
        maxDist = abs(max_edge - center)
        distance = np.max(np.vstack((minDist, maxDist)), 0)
        if all_data is None:
            all_data = distance
        else:
            all_data = np.vstack((all_data, distance))
        print(file)

    mean = np.mean(all_data, 0)
    var = np.var(all_data, 0)
    np.savetxt(mean_save, mean)
    np.savetxt(var_save, var)

if __name__ == '__main__':
    N = 1e25
    betas = get_beta_runs()
    for beta in betas:
        files = get_files(beta, N)
        mean_max_from_files(files)
        print('Finished: ' + beta + '\n')
