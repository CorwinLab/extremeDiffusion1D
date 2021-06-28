import numpy as np
import glob
import matplotlib 
matplotlib.use('Agg')
from matplotlib import pyplot as plt

files = glob.glob('/home/jhass2/Data/*/1.00e_50/max_variance.txt')
n25files = glob.glob('/home/jhass2/Data/*/1.00e_25/max_variance.txt')

def getnonzeros(files):
    betas = []
    nonzeros = []

    for file in files:
        splits = file.split('/')
        beta = splits[4]
        if beta == '0.01':
            continue
        betas.append(float(beta))
        var = np.loadtxt(file)
        first_nonzero = np.nonzero(var>0.01)[0][0]
        nonzeros.append(first_nonzero)

    return betas, nonzeros

betas, nonzeros = getnonzeros(files)
beta25, nonzeros25 = getnonzeros(n25files)
fig, ax = plt.subplots()
ax.set_xlabel('Beta')
ax.set_ylabel('First Time of Nonzero Variance/Log(N)')
ax.scatter(betas, nonzeros/np.log(float(int(1e50))), c='k', label='N=1e50') # I think I originally did int(1e50) and so it approximated 1e50 with some extra change
ax.scatter(beta25, nonzeros25/np.log(1e25), c='r', label='N=1e25')
ax.grid(True)
ax.legend()
fig.savefig('./figures/NonzeroTimes.png')
