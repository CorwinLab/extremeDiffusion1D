# @Author: Eric Corwin <ecorwin>
# @Date:   2020-08-17T11:58:36-07:00
# @Email:  eric.corwin@gmail.com
# @Filename: BarraquandCorwin.py
# @Last modified by:   ecorwin
# @Last modified time: 2020-09-04T18:24:05-07:00

import numpy as np
from numba import jit
from joblib import Parallel, delayed
import multiprocessing
from matplotlib import pyplot as plt
import time

def evolveTimeStep(occupancy, biases):
    # Motion in 1d on the positive number line
    rightShift = np.random.binomial(occupancy, biases)
    leftShift = occupancy - rightShift
    # Reflect motion that would go to the negative number line
    leftShift[1] += leftShift[0]
    # By construction these should both be zero anyway
    # We can check this by ensuring that the number of walkers stays constant
    leftShift[0] = 0
    rightShift[-1] = 0
    newOccupancy = np.zeros(occupancy.shape, dtype=occupancy.dtype)
    newOccupancy[1:] = rightShift[:-1]
    newOccupancy[:-1] += leftShift[1:]
    return newOccupancy
    # return np.roll(rightShift,1) + np.roll(leftShift,-1)

def runFixedTime(maxTime, biasFunction, systemSize=32):
    np.random.seed()
    occupancy = np.zeros(systemSize, dtype=int)
    occupancy[0] = np.iinfo(np.int64).max
    maxElement = np.zeros(maxTime, dtype=int)
    for time in range(maxTime):
        biases = biasFunction(len(occupancy))
        occupancy = evolveTimeStep(occupancy, biases)
        maxElement[time] = np.flatnonzero(occupancy)[-1]
        # Double the system size if we're at the edge
        if maxElement[time] + 1 == occupancy.shape[0]:
            occupancy = np.hstack([occupancy, np.zeros(occupancy.shape[0], dtype=int)])
    return maxElement

def einsteinBias(N):
    return np.zeros(N) + .5

def betaBias(N, alpha=1, beta=1):
    return np.random.beta(alpha, beta, size=N)

def uniformBias(N):
    return np.random.uniform(size=N)

def illEvolveTimeStep(occupancy, biases):
    # Motion in 1d on the positive number line, shifting by 1/2 step in space each time step
    rightShift = np.random.binomial(occupancy, biases)
    leftShift = occupancy - rightShift
    newOccupancy = np.zeros(occupancy.shape, dtype=occupancy.dtype)
    return leftShift + np.roll(rightShift,1)

def illRunFixedTime(maxTime, biasFunction, systemSize=None, numWalkers=None):
    #np.random.seed()
    if not systemSize:
        systemSize = maxTime + 1
    occupancy = np.zeros(systemSize, dtype=int)

    if numWalkers:
        occupancy[0] = numWalkers
    else:
        occupancy[0] = np.iinfo(np.int64).max
    history = np.zeros([systemSize, maxTime+1], dtype=int)
    biasHistory = np.zeros([systemSize, maxTime], dtype=float)
    history[:,0] = occupancy

    for time in range(maxTime):
        biases = biasFunction(len(occupancy))
        occupancy = illEvolveTimeStep(occupancy, biases)
        history[:,time+1] = occupancy
        biasHistory[:,time] = biases
    return history, biasHistory

def plotHistory(history, discrete=False, symbol='C0.', biases=None):
    systemSize, time = history.shape
    for t in range(time):
        x, y = np.meshgrid(np.arange(systemSize+1)-(t+1)/2, np.arange(t,t+2))
        if discrete == True:
            for i in range(systemSize):
                plotWalkers(history[i,t], [x[0,i], y[0,i]], symbol=symbol)
            if biases is not None:
                plt.pcolormesh(x, y, biases[:,t].reshape(systemSize,1).T, vmin=0, vmax=1, edgecolors="grey")
            else:
                plt.pcolormesh(x, y, 0*history[:,t].reshape(systemSize,1).T, vmin=0, vmax=np.sum(history[:,0]), edgecolors="grey")
        else:
            plt.pcolormesh(x, y, history[:,t].reshape(systemSize,1).T, vmin=np.min(history), vmax = np.max(history))
    plt.axis('scaled')
    plt.xlim([-time/2, time/2])

def plotWalkers(nWalkers, startPoint, symbol='C0.'):
    walkers = np.random.uniform(size=(nWalkers, 2)) * .8 + .1 + startPoint
    plt.plot(walkers[:,0], walkers[:,1], symbol, markersize=5)

@jit(nopython=True)
def numbaFloatEvolveTimeStep(occupancy, biases, smallCutoff = 1e15):
    rightShift = 0
    for i in range(len(occupancy)):
        currentValue = occupancy[i]
        # Add in the contribution from the previous cell
        occupancy[i] = rightShift
        if currentValue < smallCutoff:
            rightShift = np.random.binomial(int(currentValue), biases[i])
        else:
            rightShift = np.random.normal( loc=biases[i]*currentValue, scale = np.sqrt(currentValue) )
        occupancy[i] += currentValue - rightShift
    return occupancy

def floatEvolveTimeStep(occupancy, biases, smallCutoff = 1e15):
    # Motion in 1d on the positive number line, shifting by 1/2 step in space each time step
    small = occupancy < smallCutoff
    giant = occupancy > smallCutoff**2
    medium = np.logical_and(~small, ~giant)

    rightShift = np.zeros(shape=len(occupancy)+1)
    # If we're so large that sqrt(N) is less than the precision
    rightShift[np.hstack([False, giant])] = biases[giant] * occupancy[giant]
    # If sqrt(N) is within precision, but we're too big to use binomial then use the gaussian appx
    rightShift[np.hstack([False, medium])] = np.round( np.random.normal( loc=biases[medium]*occupancy[medium], scale = np.sqrt(occupancy[medium]) ) )
    # If we're small enough to use integer representations then use binomial
    rightShift[np.hstack([False, small])] = np.random.binomial(occupancy[small].astype(int), biases[small])

    # return occupancy - rightShift + np.roll(rightShift,1)
    return occupancy - rightShift[1:] + rightShift[:-1]

def floatRunFixedTime(maxTime, biasFunction, numWalkers=None, dtype=np.float):
    np.random.seed()
    # Start w/ the smallest system possible
    occupancy = np.zeros(2, dtype=dtype)
    origin = 0

    if numWalkers:
        occupancy[0] = numWalkers
    else:
        occupancy[0] = np.finfo(dtype).max
    edges = np.empty( (maxTime,2))

    start = time.time()
    for t in range(maxTime):
        # if np.any(np.isnan(np.sqrt(occupancy))):
        #     bad = np.isnan(np.sqrt(occupancy))
        #     print(time, occupancy[bad])
        # The origin shifts by 1/2 a step for each iteration
        origin += .5

        biases = biasFunction(len(occupancy))
        # occupancy = numbaFloatEvolveTimeStep(occupancy, biases)
        occupancy = floatEvolveTimeStep(occupancy, biases)
        # Find the filled region
        endPoints = np.where(occupancy)[0][[0,-1]]
        lenFilled = endPoints[1]-endPoints[0] + 1
        # Trim the occupancy
        newOcc = np.zeros(lenFilled + 1)
        newOcc[:lenFilled] = occupancy[endPoints[0]:(endPoints[1]+1)]
        occupancy = newOcc
        # Shift the origin
        origin -= endPoints[0]
        edges[t,:] = endPoints - origin
        if t % 10000 == 0:
            print(t)
    print(f'Finished in {time.time()-start}' )
    return edges#, occupancy

def parallelVariance(numSamples, tMax, biasFunction):
    numCores = multiprocessing.cpu_count()//2
    with Parallel( n_jobs = numCores ) as parallel:
        edges = parallel(delayed(floatRunFixedTime)(tMax, biasFunction) for i in range(numSamples))
        # results = parallel(delayed(runFixedTime)(tMax, biasFunction) for i in range(numSamples))
    return edges
    #return np.hstack(edges)


# def firstPassage(maxValue, alpha=2, beta=2):
#     occupancy = np.zeros((maxValue+1)*2, dtype=np.int64)
#     occupancy[[maxValue, maxValue+1]] = np.iinfo(np.int64).max/2
#     time = 0
#     while (occupancy[0]) == 0 and (occupancy[-1] == 0):
#         occupancy = evolveTimeStep(occupancy, np.random.beta(alpha, beta, size=occupancy.shape))
#         time += 1
#     return time, occupancy
#
# def einsteinFirstPassage(maxValue):
#     occupancy = np.zeros((maxValue+1)*2, dtype=np.int64)
#     occupancy[[maxValue, maxValue+1]] = np.iinfo(np.int64).max/2
#     time = 0
#     while (occupancy[0]) == 0 and (occupancy[-1] == 0):
#         occupancy = evolveTimeStep(occupancy, .5)
#         time += 1
#     return time, occupancy
#
# def fixedTime(maxTime, alpha=2, beta=2):
#     occupancy = np.zeros((maxTime+1)*2, dtype=np.int64)
#     occupancy[[maxTime, maxTime+1]] = np.iinfo(np.int64).max/2
#     for time in range(maxTime):
#         occupancy = evolveTimeStep(occupancy, np.random.beta(alpha, beta, size=occupancy.shape))
#     return np.max(np.abs(np.nonzero(occupancy)[0]-(maxTime+.5)) - .5).astype(np.int64), occupancy
#
# def einsteinFixedTime(maxTime):
#     occupancy = np.zeros((maxTime+1)*2, dtype=np.int64)
#     occupancy[[maxTime, maxTime+1]] = np.iinfo(np.int64).max/2
#     for time in range(maxTime):
#         occupancy = evolveTimeStep(occupancy, .5)
#     return np.max(np.abs(np.nonzero(occupancy)[0]-(maxTime+.5)) - .5).astype(np.int64), occupancy
#
# def width(timeRange, numSamples=30):
#     returnData = []
#     mE = np.zeros(numSamples)
#     mBC = np.zeros(numSamples)
#     for t in timeRange:
#         for i in range(numSamples):
#             mE[i],_ = einsteinFixedTime(t)
#             mBC[i],_ = fixedTime(t)
#         returnData.append([np.mean(mE), np.var(mE), np.mean(mBC), np.var(mBC)])
#         print(t, returnData[-1])
#
#     return np.array(returnData)
