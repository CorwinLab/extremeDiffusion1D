import numpy as np
import glob
import pandas as pd
from matplotlib import pyplot as plt
import sys 
sys.path.append("../../dataAnalysis")
from theory import KPZ_var_fit

def prefactor(x, N):
    logN = np.log(N)
    return 1/2 * (1 + logN / x)

def var_theory(x, N):
    logN = np.log(N).astype(float)
    return x**4 / 4 / logN**4 * KPZ_var_fit(8 * logN**3 / x**2) #* prefactor(x, N)

def I(v): 
    return 1-np.sqrt(1-v**2)

def sigma(v): 
    return (2 * I(v)**2 / (1-I(v)))**(1/3)

def var_short(x, N):
    logN = np.log(N).astype(float)
    t0 = (logN**2 + x**2)/ (2*logN)
    return (t0**(1/3) * sigma(x/t0) / (I(x/t0) - x**2 / t0**2 / np.sqrt(1-(x/t0)**2)))**2 * 0.8133 #* prefactor(x, N)
    
dir = "/home/jacob/Desktop/talapasMount/JacobData/FPTDoubleSided/F*.txt"
files = glob.glob(dir)

df_tot = pd.DataFrame()
for f in files:
    df = pd.read_csv(f)
    df.drop(df[df["Distance"] == "Distance"].index, inplace=True)
    df_tot = df_tot.append(df, ignore_index=True)

df_tot.reset_index(inplace=True, drop=True)
df_tot['Distance'] = df_tot['Distance'].values.astype(int)
df_tot['Time'] = df_tot['Time'].values.astype(int)
df_tot.to_csv('Total_Data.csv')

distances = np.unique(df_tot['Distance'].values)

vars = []
means = []
for d in distances:
    data = df_tot[df_tot['Distance'] == d]
    vars.append(np.var(data['Time'].values))
    means.append(np.mean(data["Time"].values))

N = 1e7
logN = np.log(N)
theoretical_var_short = var_short(distances, N)
theoretical_var_long = var_theory(distances, N)

fig, ax = plt.subplots()
ax.plot(distances / logN, vars)
ax.plot(distances / logN, theoretical_var_short)
ax.plot(distances / logN, theoretical_var_long)
ax.set_xscale("log")
ax.set_yscale("log")
fig.savefig("Variance.png", bbox_inches='tight')

fig, ax = plt.subplots()
ax.plot(distances / logN, means)
ax.set_xscale("log")
ax.set_yscale("log")
fig.savefig("Mean.png", bbox_inches='tight')
