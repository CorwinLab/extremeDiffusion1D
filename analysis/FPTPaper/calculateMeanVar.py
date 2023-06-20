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
        if N != 28:
            continue
        dir = home_dir + f'/{N}/Q*.txt'
        files = glob.glob(dir)
        df, number_of_files = calculateMeanVarDiscrete(files, max_dist, verbose=True, nFile=2500)
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

einstein_files = glob.glob('/home/jacob/Desktop/talapasMount/JacobData/FPTDiscreteE/12/Q*.txt')
max_dist = 20721
recalculate_mean = False
if recalculate_mean: 
    df, number_of_files = calculateMeanVarDiscrete(einstein_files, max_dist)
    df.to_csv('/home/jacob/Desktop/talapasMount/JacobData/FPTDiscreteE/12/MeanVariance.csv', index=False)
    np.savetxt('/home/jacob/Desktop/talapasMount/JacobData/FPTDiscreteE/12/NumberOfSystems.csv', [number_of_files])
    
num_files = np.loadtxt('/home/jacob/Desktop/talapasMount/JacobData/FPTDiscreteE/12/NumberOfSystems.csv')
print(f"Einstein Discrete: {num_files} files")

talapas_dir = "/home/jacob/Desktop/talapasMount/JacobData/CleanData/FPTCDFPaperFixed"
dirs = os.listdir(talapas_dir)
maxFiles = [2000, 2000, 2000, 2000, 1000]
recalculate_mean = False
if recalculate_mean: 
    nFiles = []
    for max_dist, N, nFile in zip(max_dists, Ns, maxFiles):
        if N != 28:
            continue
        talapas_file_dir = talapas_dir + f'/{N}/First*.txt'
        files = glob.glob(talapas_file_dir)
        df, number_of_files = calculateMeanVarCDF(files, max_dist, verbose=True, nFile=nFile)
        path = os.path.join(talapas_dir, f'{N}', 'MeanVariance.csv')
        df.to_csv(path, index=False)
        nFiles.append(number_of_files)
        print(f"CDF {N}: {number_of_files} files")
        print("Fixed saving to:", path)
        print(f"Directories used: {talapas_dir}")

        np.savetxt(os.path.join(talapas_dir, f'{N}', 'NumberOfSystems.txt'), [number_of_files])

for N in Ns: 
    num_files = np.loadtxt(os.path.join(talapas_dir, f'{N}', 'NumberOfSystems.txt'))
    print(f"CDF {N}: {num_files} files")