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

def runC(num_of_steps, N, smallCutoff):
    occ = np.zeros(num_of_steps)
    occ[0] = int(N)

    d = cdiff.Diffusion(int(N), 1, smallCutoff)
    d.setOccupancy(occ)
    d.evolveTimesteps(num_of_steps)

cevolveTimesteps = timer(runC, N=3)
floatRunFixedTime = timer(diff.floatRunFixedTime, N=3)

Ns = [int(1e5), (1e10), (1e15), (1e25), (1e35), 1e50]
beta = 1.0

ctimes = []
pytimes = []
for N in Ns:
    logN = np.log(N)
    smallCutoff = int(1e9)
    num_of_steps = logN ** (5/2)
    num_of_steps = round(num_of_steps)

    py_times = floatRunFixedTime(num_of_steps, diff.betaBias, N)

    c_times = cevolveTimesteps(num_of_steps, N, smallCutoff)

    scientificN = "{:e}".format(N)
    np.savetxt(f'./times/ctimes{scientificN}.txt', c_times)
    np.savetxt(f'./times/pytimes{scientificN}.txt', py_times)
    ctimes.append(np.mean(c_times))
    pytimes.append(np.mean(py_times))
    print(f'Finished N={scientificN}')

fig, ax = plt.subplots()
ax.scatter(Ns, ctimes, c='k', label='C++')
ax.scatter(Ns, pytimes, c='b', label='Python')
ax.set_xlabel('Number of particles/Iterations')
ax.set_ylabel('Time to run (sec)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()
fig.savefig('Speed.png')
