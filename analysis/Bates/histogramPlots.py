from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np
import pandas as pd
import glob
import sys 
import copy
import matplotlib
sys.path.append("../../dataAnalysis")
from overalldatabase import Database

path = "/home/jacob/Desktop/corwinLabMount/CleanData/DeltaBeta01/Q*.txt"
files = glob.glob(path)
run_again = False

N=24
db = Database()
path = "/home/jacob/Desktop/corwinLabMount/CleanData/DeltaBeta01/"
db.add_directory(path, dir_type="Gumbel")
#db.calculateMeanVar(path, verbose=True, maxTime=55262)
cdf_df, max_df = db.getMeanVarN(N)

if run_again:
    concat_df = pd.DataFrame()
    for i, f in enumerate(files):
        data = pd.read_csv(f)
        time = data['time']
        Nquant = data['1000000000000000000000000']
        concat_df['time'] = time
        concat_df[f'quantile{i}'] = Nquant
        print(i/len(files) * 100)

    concat_df.to_csv("Concat.csv")
else: 
    concat_df = pd.read_csv("Concat.csv")

# Make intensity plot
data = concat_df.values[:, 2:]
bins = np.arange(-250, 250, step=2)
intensity_values = []
for row in range(data.shape[0]):
    mean_centered = data[row, :] - np.mean(data[row, :])
    hist_values, _ = np.histogram(data[row, :]-np.mean(data[row, :]), bins=bins)
    assert sum(hist_values) == 5001, print(sum(hist_values), row)
    intensity_values.append(hist_values)

fig, ax = plt.subplots()
ax.imshow(np.array(intensity_values).T, 'Greys_r', vmin=1, vmax=2)
fig.savefig("Intensity.png")

run_again = False
if run_again: 
    min_data = -250
    max_data = 250
    nbins = 100
    bins = np.linspace(min_data, max_data, nbins)
    logN = np.log(float(f"1e{N}"))
    intensity = np.zeros(bins.shape[0]-1)
    # Make gif of histograms
    for row in range(0, data.shape[0], 15):
        fig, (ax, ax1, ax2) = plt.subplots(nrows=3, figsize=(6, 10))

        time = concat_df['time'].iloc[row]
        mean_centered = data[row, :] - np.mean(data[row, :])
        n, bins, patches = ax.hist(mean_centered, bins=bins, label=f't={time}')
        intensity = np.vstack((intensity, n))
        ax.set_xlim([min_data, max_data])
        ax.set_xlabel("Distance")
        ax.set_ylabel("Counts")
        ax.legend()

        ax2.plot(cdf_df['time'] / logN, cdf_df['Mean Quantile'])
        ax2.set_xlabel("Time / log(N)")
        ax2.set_ylabel("Mean(Max)")
        ax2.scatter(time / logN, np.mean(data[row, :]), c='r')
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlim([1 / logN, max(cdf_df['time']) / logN])

        ax1.plot(cdf_df['time'] / logN, cdf_df['Var Quantile'])
        ax1.scatter(time / logN, np.var(data[row, :]), c='r')
        ax1.set_ylabel("Var(Max)")
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_xlim([1 / logN, max(cdf_df['time']) / logN])

        plt.tight_layout()
        #fig.savefig(f"./Histograms/Histogram{row}.png", dpi=100)
        plt.close(fig)
        print(row / data.shape[0] * 100)

    np.savetxt("Intensity.txt", intensity)

else: 
    intensity = np.loadtxt("Intensity.txt")

cmap = copy.copy(matplotlib.cm.get_cmap("rainbow"))
cmap.set_under(color="white")
cmap.set_bad(color="white")

fig, ax = plt.subplots()
cax = ax.imshow(intensity.T[:, 1:], cmap=cmap, norm=colors.LogNorm(vmin=1, vmax=5000), interpolation=None)
ax.set_xlabel("Time")
ax.set_ylabel("Centered Maximum Particle Location")
ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelbottom=False, labelleft=False)
fig.savefig("Intensity2.pdf", bbox_inches="tight")