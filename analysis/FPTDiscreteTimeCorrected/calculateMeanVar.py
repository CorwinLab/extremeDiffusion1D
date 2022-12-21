import numpy as np
import os 
import sys
sys.path.append("../../dataAnalysis")
from FPTDataAnalysis import calculateMeanVarDiscrete, calculateMeanVarCDF
import glob
import pandas as pd

Ns = [1, 2, 5, 12, 28]
N_vals = [float(f"1e{i}") for i in Ns]
max_dists = [1723, 3436, 8531, 20461, 47967]

home_dir = '/home/jacob/Desktop/talapasMount/JacobData/FPTDiscreteTimeCorrected'
dirs = os.listdir(home_dir)
recalculate_mean = False
if recalculate_mean:
    nFiles = []
    for max_dist, N in zip(max_dists, Ns):
        dir = home_dir + f'/{N}/Q*.txt'
        files = glob.glob(dir)
        df, number_of_files = calculateMeanVarDiscrete(files, max_dist, verbose=True)
        if df is None: 
            continue
        path = os.path.join(home_dir,f'{N}', f'MeanVariance.csv')
        df.to_csv(path, index=False)
        nFiles.append(number_of_files)
        print(f"Max {N}: {number_of_files} files")

        np.savetxt(home_dir + f'/{N}/NumberOfSystems.txt', [number_of_files])

for N in Ns: 
    num_files = np.loadtxt(home_dir + f'/{N}/NumberOfSystems.txt')
    print(f"Discrete {N}: {num_files} files")


cdf_dir = '/home/jacob/Desktop/corwinLabMount/CleanData/FPTCDFPaper'
talapas_dir = '/home/jacob/Desktop/corwinLabMount/CleanData/TalapasFPTCDF/FPTCDFPaper'
dirs = os.listdir(cdf_dir)
recalculate_mean = False
if recalculate_mean: 
    nFiles = []
    for max_dist, N in zip(max_dists, Ns):
        dir = cdf_dir + f'/{N}/First*.txt'
        talapas_file_dir = talapas_dir + f'/{N}/First*.txt'
        talapas_files = glob.glob(talapas_file_dir)
        files = glob.glob(dir) + talapas_files
        df, number_of_files = calculateMeanVarCDF(files, max_dist, verbose=False)
        path = os.path.join(cdf_dir,f'{N}', 'MeanVariance.csv')
        df.to_csv(path, index=False)
        nFiles.append(number_of_files)
        print(f"CDF {N}: {number_of_files} files")

        np.savetxt(os.path.join(cdf_dir, f'{N}', 'NumberOfSystems.txt'), [number_of_files])

for N in Ns: 
    num_files = np.loadtxt(os.path.join(cdf_dir, f'{N}', 'NumberOfSystems.txt'))
    print(f"CDF {N}: {num_files} files")

einstein_files = glob.glob('/home/jacob/Desktop/talapasMount/JacobData/FPTDiscreteE/12/Q*.txt')
max_dist = 20721
recalculate_mean = False
if recalculate_mean: 
    df, number_of_files = calculateMeanVarDiscrete(einstein_files, max_dist)
    df.to_csv('/home/jacob/Desktop/talapasMount/JacobData/FPTDiscreteE/12/MeanVariance.csv', index=False)
    np.savetxt('/home/jacob/Desktop/talapasMount/JacobData/FPTDiscreteE/12/NumberOfSystems.csv', [number_of_files])
    
num_files = np.loadtxt('/home/jacob/Desktop/talapasMount/JacobData/FPTDiscreteE/12/NumberOfSystems.csv')
print(f"Einstein Discrete: {num_files} files")
