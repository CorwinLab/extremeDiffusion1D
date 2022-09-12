import glob 
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

dir = "/home/jacob/Desktop/corwinLabMount/CleanData/FirstPassTest/F*.csv"
files = glob.glob(dir)
df_tot = pd.DataFrame()
for f in files:
    df = pd.read_csv(f)
    df_tot = df_tot.append(df, ignore_index=True)

df_tot.reset_index(inplace=True, drop=True)
unique_distances = np.unique(df_tot['Distances'].values)

vars = []
means = []
for d in unique_distances:
    df_dist = df_tot[df_tot['Distances'] == d]
    if d == 10:
        print(np.unique(df_dist[df_dist['Side'] == 'right']['Time']))
    var = np.var(df_dist['Time'].values)
    vars.append(var)
    mean = np.mean(df_dist['Time'].values)
    means.append(mean)

logN = np.log(1e7)

theoretical_distance7 = np.loadtxt("../FixedFirstPassCDF/distances7.txt")
theoretical_variance7 = np.loadtxt("../FixedFirstPassCDF/varianceShortTime7.txt")
theoretical_variance7_long = np.loadtxt("../FixedFirstPassCDF/varianceLongTime7.txt")

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.plot(unique_distances / logN, vars)
ax.plot(theoretical_distance7 / logN, theoretical_variance7)
ax.plot(theoretical_distance7 / logN, theoretical_variance7_long)
fig.savefig("Variance.png", bbox_inches='tight')

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.plot(unique_distances / logN, means)
fig.savefig("Mean.png", bbox_inches='tight')