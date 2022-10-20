import glob 
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import sys 
sys.path.append("../../dataAnalysis")
from theory import KPZ_var_fit

dir = "/home/jacob/Desktop/corwinLabMount/CleanData/FirstPassTest/F*.csv"
files = glob.glob(dir)
df_tot = pd.DataFrame()
for f in files:
    df = pd.read_csv(f)
    df_tot = df_tot.append(df, ignore_index=True)

df_tot.reset_index(inplace=True, drop=True)
df_tot.to_csv('Total_Times.csv')
unique_distances = np.unique(df_tot['Distances'].values)

vars = []
means = []
for d in unique_distances:
    df_dist = df_tot[df_tot['Distances'] == d]
    var = np.var(df_dist['Time'].values)
    vars.append(var)
    mean = np.mean(df_dist['Time'].values)
    means.append(mean)

vars_min = []
means_min = []
for d in unique_distances:
    df_dist = df_tot[df_tot['Distances'] == d]
    df_dist = df_dist[df_dist['Number Crossed'] == 0]
    df_dist = df_dist[df_dist['Side'] == 'right']
    if d == 10:
        print(df_dist)
        print(np.unique(df_dist['Time'], return_counts=True))
    var = np.var(df_dist['Time'].values)
    vars_min.append(var)
    mean = np.mean(df_dist['Time'].values)
    means_min.append(mean)

N = 1e7
logN = np.log(N)

def var_theory(x, N):
    logN = np.log(N).astype(float)
    return x**4 / 4 / logN**4 * KPZ_var_fit(8 * logN**3 / x**2)

def I(v): 
    return 1-np.sqrt(1-v**2)

def sigma(v): 
    return (2 * I(v)**2 / (1-I(v)))**(1/3)

def var_short(x, N):
    logN = np.log(N).astype(float)
    t0 = (logN**2 + x**2)/ (2*logN)
    return (t0**(1/3) * sigma(x/t0) / (I(x/t0) - x**2 / t0**2 / np.sqrt(1-(x/t0)**2)))**2 * 0.8133

theory_dist = unique_distances[unique_distances > logN]
theoretical_variance_short = var_short(theory_dist, N)
theoretical_variance_long = var_theory(theory_dist, N)

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.plot(unique_distances / logN, vars)
ax.plot(theory_dist / logN, theoretical_variance_short, '--')
ax.plot(theory_dist / logN, theoretical_variance_long, '--')
ax.plot(unique_distances / logN, vars_min)
fig.savefig("Variance.png", bbox_inches='tight')

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.plot(unique_distances / logN, means)
fig.savefig("Mean.png", bbox_inches='tight')