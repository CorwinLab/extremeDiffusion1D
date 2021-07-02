import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import glob

files = glob.glob('/home/jhass2/Data/1.0/1/Q*.txt')
t = 0
NsHist = []

for f in files: 
    data = np.loadtxt(f)
    time = data[:, 0]
    N50 = data[:, -1]
    idx = len(time)//2
    t = time[idx]
    NsHist.append(N50[idx])

N = 1e50
t = int(t)
tstr = '{:.0e}'.format(t)
fig, ax = plt.subplots()
ax.hist(NsHist)
ax.set_xlabel('Nth Quartile')
ax.set_ylabel('Count')
ax.set_title(f'Histogram at t={tstr}')
fig.savefig('NthQuartileHistogram'+tstr+'.png')
