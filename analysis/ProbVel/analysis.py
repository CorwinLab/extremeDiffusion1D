import sys
sys.path.append("../../src")

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import npquad
import json
import os
import glob
import pandas as pd
from TracyWidom import TracyWidom

tw = TracyWidom(beta=2)
x = np.linspace(-5, 5, 100)
pdf = tw.pdf(x)

def I(v):
    return 1 - np.sqrt(1-v**2)

def sigma(v):
    return (2*I(v)**2/(1-I(v)))**(1/3)

N = np.quad("1e50")

directory = "/home/jacob/Desktop/talapasMount/JacobData/ProbVel/P*.txt"
files = glob.glob(directory)

df = pd.DataFrame(columns=['time', 'prob', 'vel'])
print(len(files))

for f in files:
    try:
        data = np.loadtxt(f, skiprows=1, delimiter=',')
        data_df = pd.DataFrame(data, columns=['time', 'prob', 'vel'])
        df = pd.concat([df, data_df])
    except:
        continue
df.reset_index(inplace=True, drop=True)
df.to_csv('Data.txt', index=False)
times = np.unique(df['time'])

for t in times:
    df_time = df[df['time'] == t]
    prob = df_time['prob'].values
    v = df_time['vel'].values

    fig, ax = plt.subplots()
    chi_val = (np.log(prob) + t * I(v)) / (sigma(v) * t**(1/3))
    ax.hist(chi_val, bins=50, density=True)
    ax.plot(x, pdf)
    ax.set_yscale("log")
    fig.savefig(f"Histogram{t}.png", bbox_inches='tight')
