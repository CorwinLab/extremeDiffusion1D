import numpy as np
import time
import sys

sys.path.append('../src')

import cdiffusion as cdiff
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

cfloatEvolveTimestep = timer(cdiff.iterate_timestep)
floatEvolveTimeStep = timer(diff.floatEvolveTimeStep)

biases = np.random.random(500000)
occ = np.random.random(499999) * 100
occ = np.append(occ, 0)
occ = np.round(occ)
c_occ = cfloatEvolveTimestep(occ, biases, smallcutoff=1000)
reg_occ = floatEvolveTimeStep(occ, biases, smallCutoff=1000)

print(np.mean(c_occ))
print(np.mean(reg_occ))
