import numpy as np
import npquad 
import json 
from matplotlib import pyplot as plt
import matplotlib as mpl

file = '/home/jacob/Desktop/talapasMount/JacobData/LongFirstPassCDF24/Scalars0.json'
f = open(file)
data = json.load(f)
pdfs = data['pdfsData']
positions = pdfs.keys()
fig, ax = plt.subplots()
ax.set_yscale("log")

cmap = mpl.cm.get_cmap("Spectral")
colors = [cmap(1.0 * i / len(positions) / 1) for i in range(len(positions))]

for i, maxPosition in enumerate(positions):
    max_pdf = pdfs[maxPosition]['pdf']
    max_pdf = [float(num) for num in max_pdf]
    maxPosition = int(maxPosition)
    xvals = np.arange(-int(maxPosition)-2, int(maxPosition)+2, 2)
    ax.plot(xvals, max_pdf, c=colors[i])

fig.savefig(f"PDF.png")

positions = [int(i) for i in positions]
maxPosition = str(max(positions))
print("time:", data['pdfsData'][str(maxPosition)]['time'])
max_pdf = pdfs[maxPosition]['pdf']
max_pdf = [float(num) for num in max_pdf]
print("Position:", maxPosition)
xvals = np.arange(-int(maxPosition)-2, int(maxPosition)+2, 2)
fig, ax = plt.subplots()
ax.set_yscale("log")
ax.plot(xvals, max_pdf, c=colors[i])
ax.plot(-xvals, max_pdf)
fig.savefig(f"MaxPDF.png")