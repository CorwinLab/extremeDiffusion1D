import numpy as np
from matplotlib import pyplot as plt
import glob
import pandas as pd

dirs = ['/home/jacob/Desktop/talapasMount/JacobData/20/F*.csv',
        '/home/jacob/Desktop/talapasMount/JacobData/100/F*.csv',
        '/home/jacob/Desktop/talapasMount/JacobData/FirstPassTest/F*.csv']
distances = [20, 100, 1611]
for d, dir in zip(distances, dirs):
    print(dir)
    files = glob.glob(dir)

    df_tot = pd.DataFrame()
    for f in files:
        try:
            df = pd.read_csv(f)
        except:
            continue
        df_tot = pd.concat([df_tot, df], ignore_index=True)

    print(df_tot)
    df_tot.reset_index(inplace=True, drop=True)
    df_tot.to_csv(f"TotalDF{d}.csv", index=False)
    df_single = df_tot[df_tot['Number Crossed'] == 0]
    nBins = 100
    bins = np.linspace(min(df_tot['Time']), max(df_tot['Time']), nBins)

    fig, ax = plt.subplots()
    ax.hist(df_tot['Time'], bins=bins, alpha=0.5, density=True)
    ax.hist(df_single['Time'], bins=bins, alpha=0.5, density=True)
    ax.set_xlabel("Time")
    ax.set_ylabel("Probability Density")
    fig.savefig(f"Histogram{d}.pdf", bbox_inches='tight')
