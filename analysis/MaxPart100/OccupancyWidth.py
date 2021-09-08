from matplotlib import pyplot as plt
import numpy as np

occupancy = np.loadtxt("/home/jacob/Desktop/corwinLabMount/CleanData/MaxPart100/Occupancy18.txt", delimiter=",")
nonzeros = np.argwhere(occupancy)
minidx, maxidx = (min(nonzeros)[0], max(nonzeros)[0])
width = maxidx - minidx
fig, ax = plt.subplots()
ax.scatter(range(minidx, maxidx), occupancy[minidx:maxidx], s=2, label=f'Width={width}')
ax.set_xlabel("Position")
ax.set_ylabel("Number of Particles")
ax.legend()
fig.savefig("occupancy.png")
