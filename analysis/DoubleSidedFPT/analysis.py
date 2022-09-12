import numpy as np
import glob
import pandas as pd
from matplotlib import pyplot as plt

dirs = ['/home/jacob/Desktop/talapasMount/JacobData/DoubleSidedFPT/100/F*.csv']
distances = [100]
for d, dir in zip(distances, dirs):
    files = glob.glob(dir)

    df_tot = pd.DataFrame()
    min_times = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except:
            continue
        min_times.append(min(df['Time']))
        df_tot = pd.concat([df_tot, df], ignore_index=True)

    df_tot.reset_index(inplace=True, drop=True)
    df_tot.to_csv(f"TotalDF{d}.csv", index=False)
    df_single = df_tot[df_tot['Number Crossed'] == 0]
    nBins = 100
    bins = np.linspace(min(df_tot['Time']), max(df_tot['Time']), nBins)

    fig, ax = plt.subplots()
    ax.hist(df_tot['Time'], bins=bins, alpha=0.5, density=True, label='All FPT')
    ax.hist(min_times, bins=bins, alpha=0.5, density=True, label='Double-sided 1/N FPT')
    ax.hist(df_tot[df_tot['Side']=='left']['Time'], bins=bins, alpha=0.5, density=True, label='Left')
    ax.hist(df_tot[df_tot['Side']=='right']['Time'], bins=bins, alpha=0.5, density=True, label='Right')
    ax.set_xlabel("Time")
    ax.set_ylabel("Probability Density")
    ax.legend()
    fig.savefig(f"Histogram{d}.pdf", bbox_inches='tight')
    np.savetxt("MinimumTimes.txt", min_times)