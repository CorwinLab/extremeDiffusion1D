import numpy as np
import time
import sys
from matplotlib import pyplot as plt

sys.path.append('../src')
sys.path.append('../cDiffusion')

import cDiffusion as cdiff
import diffusion as diff

def timer(func, N=10):
    '''
    Decorator to run a function N number of times and return the times of each
    run.
    '''
    def wrapper(*args, **kwargs):
        '''
        Return list of how long it took to run the function for each call.
        '''
        times = []
        for _ in range(N):
            start = time.time()
            output = func(*args, **kwargs)
            t = time.time() - start
            times.append(t)
        return times

    return wrapper

cevolveTimesteps = timer(cdiff.evolveTimesteps, N=3)
floatRunFixedTime = timer(diff.floatRunFixedTime, N=3)

Ns = [int(1e2), int(1e3), int(1e4), int(1e5)]
beta = 1.0
smallCutoff = int(1e15)

ctimes = []
pytimes = []
for N in Ns:
    c_occ = cevolveTimesteps(N, beta, smallCutoff)
    reg_occ = floatRunFixedTime(N, diff.betaBias, N)
    ctimes.append(np.mean(c_occ))
    pytimes.append(np.mean(reg_occ))
    print(f'Finished N={N}')

fig, ax = plt.subplots()
ax.scatter(Ns, ctimes, c='k', label='C++')
ax.scatter(Ns, pytimes, c='b', label='Python')
ax.set_xlabel('Number of particles/Iterations')
ax.set_ylabel('Time to run (sec)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()
fig.savefig('Speed.png')
