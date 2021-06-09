# @Author: Eric Corwin <ecorwin>
# @Date:   2020-08-17T11:58:36-07:00
# @Email:  eric.corwin@gmail.com
# @Filename: BarraquandCorwin.py
# @Last modified by:   ecorwin
# @Last modified time: 2021-01-18T13:24:49-08:00

import numpy as np

def einsteinBias(N):
    return np.zeros(N) + .5

def betaBias(N, alpha=1, beta=1):
    return np.random.beta(alpha, beta, size=N)

def uniformBias(N):
    return np.random.uniform(size=N)

def floatEvolveTimeStep(occupancy, biases, smallCutoff = 1e15):
    '''
    args:
    occupancy (array of floats): How many walkers are at each element
    biases (array of floats): Weight of our weighted coin at each element for this timestep
    smallCutoff (float): The precision of our floating point number
    '''
    # Motion in 1d on the positive number line, shifting by 1/2 step in space each time step
    # Small numbers can be treated like an integer using binomial
    small = occupancy < smallCutoff
    # Giant numbers we don't need to worry about variance since it won't matter anyway
    giant = occupancy > smallCutoff**2
    # Medium numbers we can use the gaussian approximation
    medium = np.logical_and(~small, ~giant)

    rightShift = np.zeros(shape=len(occupancy)+1)
    # If we're so large that sqrt(N) is less than the precision
    rightShift[np.hstack([False, giant])] = np.round(biases[giant] * occupancy[giant])
    # If sqrt(N) is within precision, but we're too big to use binomial then use the gaussian appx
    mediumVariance = occupancy[medium] * biases[medium] * (1-biases[medium])
    rightShift[np.hstack([False, medium])] = np.ceil(np.random.normal( loc=biases[medium]*occupancy[medium], scale = np.sqrt(mediumVariance) ))
    # If we're small enough to use integer representations then use binomial
    rightShift[np.hstack([False, small])] = np.random.binomial(occupancy[small].astype(int), biases[small])

    # return occupancy - rightShift + np.roll(rightShift,1)
    returnData = np.round(occupancy - rightShift[1:] + rightShift[:-1])
    # enforce that the returnData is always positive
    returnData[returnData < 0] = 0
    if np.any(returnData < 0):
        neg = np.where(returnData < 0)[0][0]
        print(returnData[neg-5:neg+5])
        print(occupancy[neg-5:neg+5])
        right = rightShift[1:]
        left = rightShift[:-1]
        print(-right[neg-5:neg+5] + left[neg-5:neg+5])

    return returnData #np.round(occupancy + (- rightShift[1:] + rightShift[:-1]))

def floatRunFixedTime(maxTime, biasFunction, numWalkers=None, dtype=np.float):
    # This is useful for running things in parallel
    np.random.seed()
    # Start w/ the smallest system possible
    occupancy = np.zeros(2, dtype=dtype)
    origin = 0

    if numWalkers:
        occupancy[0] = numWalkers
    else:
        occupancy[0] = np.finfo(dtype).max

    edges = np.empty( (maxTime,2) )
    # start = time.time()

    for t in range(maxTime):
        # The origin shifts by 1/2 a step for each iteration
        origin += .5

        biases = biasFunction(len(occupancy))
        # occupancy = numbaFloatEvolveTimeStep(occupancy, biases)
        occupancy = floatEvolveTimeStep(occupancy, biases)
        # Find the filled region, the first index tells us to extract the list from what where gives us
        # The second list gives us the first [0] and last [-1] element
        endPoints = np.where(occupancy)[0][[0,-1]]
        lenFilled = endPoints[1]-endPoints[0] + 1
        # Trim the occupancy
        newOcc = np.zeros(lenFilled + 1)
        newOcc[:lenFilled] = occupancy[endPoints[0]:(endPoints[1]+1)]
        occupancy = newOcc
        edges[t,:] = endPoints - origin
        # Shift the origin
        origin -= endPoints[0]

        # if t % 10000 == 0:
        #     print(t)
    # print('Finished in', time.time()-start )
    return edges#, occupancy
