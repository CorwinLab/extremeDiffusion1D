import numpy as np
from numba import jit
from matplotlib import pyplot as plt
import time

@jit(nopython=True)
def numbaDirichlet(numSites, shape):
    a = np.ones( (shape[0], shape[1], numSites+1) )
    for i in range(shape[0]):
        for j in range(shape[1]):
            a[i,j,0] = 0
            a[i,j,1:numSites] = np.sort(np.random.rand(numSites-1))
    return np.diff(a)
    #return np.append(2,3)
    #return np.diff(np.hstack( ( np.sort(np.random.rand(3)), np.array(1) )))

# @jit(nopython=True)
def twoDBCModelPDF(occupancy, tMax):

    for t in range(1,tMax+1):
        newOccupancy = np.zeros(occupancy.shape, dtype=occupancy.dtype)
        # bias = .25*np.ones([4,t,t])
        # bias = np.moveaxis( np.random.dirichlet(4*[1], [t, t]), 2, 0)
        bias = np.random.dirichlet(4*[1], [t, t])
        # bias = numbaDirichlet(4, [t,t])
        newOccupancy[:t   ,:t   ] += bias[:,:,0] * occupancy[:t,:t]
        newOccupancy[1:t+1,:t   ] += bias[:,:,1] * occupancy[:t,:t]
        newOccupancy[:t   ,1:t+1] += bias[:,:,2] * occupancy[:t,:t]
        newOccupancy[1:t+1,1:t+1] += bias[:,:,3] * occupancy[:t,:t]
        occupancy = newOccupancy

    return occupancy
