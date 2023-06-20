import sys

sys.path.append("../dataAnalysis/")
from pyDiffusion import DiffusionPDF
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np
import copy

N = 100_000
numSteps = 1000
numSteps = int(numSteps)
d = DiffusionPDF(N, 'beta', [1, 1], numSteps, ProbDistFlag=False, staticEnvironment=False)
allOcc = np.zeros(shape=(numSteps + 1, numSteps + 1))
times = np.arange(0, numSteps+1)
maxPosition = np.zeros(numSteps+1)

for i in range(numSteps):
    d.iterateTimestep()
    occ = d.getOccupancy()
    occ = np.array(occ, dtype=np.float64)
    allOcc[i, :] = occ
    maxPos = 2 * d.edges[1] - d.currentTime
    maxPosition[i] = maxPos

for i in range(allOcc.shape[0]):
    occ = allOcc[i, :]
    idx_shift = int((max(d.center) - d.center[i]))
    occ = np.roll(occ, idx_shift)
    allOcc[i, :] = occ
    
d = DiffusionPDF(N, 'beta', [float("inf"), float("inf")], numSteps, ProbDistFlag=False, staticEnvironment=False)
allOccE = np.zeros(shape=(numSteps + 1, numSteps + 1))
maxPositionE = np.zeros(numSteps+1)

for i in range(numSteps):
    d.iterateTimestep()
    occ = d.getOccupancy()
    occ = np.array(occ, dtype=np.float64)
    allOccE[i, :] = occ
    maxPos = 2 * d.edges[1] - d.currentTime
    maxPositionE[i] = maxPos

for i in range(allOccE.shape[0]):
    occ = allOccE[i, :]
    idx_shift = int((max(d.center) - d.center[i]))
    occ = np.roll(occ, idx_shift)
    allOccE[i, :] = occ
    
color = 'tab:red'
cmap = copy.copy(matplotlib.cm.get_cmap('rainbow'))
cmap.set_under(color='white')
cmap.set_bad(color='white')
vmax = N
vmin = 0.00001
times = allOcc.T.shape[0]+10 - np.unique(np.geomspace(allOcc.T.shape[0], 10, num=75).astype(int))
for t in times:
    fig, ax = plt.subplots()
    allOccPlot = copy.copy(allOcc.T)
    print(t)
    allOccPlot[:, t:] = 0
    cax = ax.imshow(
        allOccPlot,
        norm=colors.LogNorm(vmin=1, vmax=vmax),
        cmap=cmap,
        aspect="auto",
        interpolation="none",
    )

    # Plot the RWRE Occupation
    #ax.set_ylabel("Distance")
    #ax.set_yticks(np.linspace(0, allOcc.shape[1], 21))
    #ticks = ax.get_yticks()
    #new_ticks = np.linspace(0, allOcc.shape[1], len(ticks)) - (allOcc.shape[1]) / 2
    #new_ticks = list(new_ticks.astype(int))
    #ax.set_yticklabels([])
    #ax.set_xticklabels([])
    ax.axis("off")
    dist = 200
    ax.set_ylim([(allOcc.shape[1]) / 2 - dist - 1, (allOcc.shape[1]) / 2 + dist + 1])

    # This is some bullshit rescaling because the matplotlib image origin sucks
    # plot_pos = maxPosition/2 + 500
    # ax.plot(times[:-1], plot_pos[:-1], c='r', zorder=2, lw=2, label=r'$\mathrm{Max}^N_t$')

    # leg = ax.legend(
    #     loc="upper right",
    #     framealpha=0,
    #     labelcolor='r',
    #     handlelength=0,
    #     handletextpad=0,
    # )
    # for item in leg.legendHandles:
    #     item.set_visible(False)

    fig.savefig(f"./Video/Occupation{t}.png", bbox_inches="tight")
    plt.close(fig)