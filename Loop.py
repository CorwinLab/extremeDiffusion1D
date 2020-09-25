# @Author: Eric Corwin <ecorwin>
# @Date:   2020-09-24T15:48:46-07:00
# @Email:  eric.corwin@gmail.com
# @Filename: Loop.py
# @Last modified by:   ecorwin
# @Last modified time: 2020-09-25T13:49:33-07:00

import BarraquandCorwin as bc
import numpy as np

def computeEdgeVariance(minNumWalkers = 1e10, maxNumWalkers = 1e300, numRuns = 100):

    # Initial number of particles
    numSteps = np.log10(maxNumWalkers/minNumWalkers)/10 + 1
    numWalkersList = np.geomspace(minNumWalkers, maxNumWalkers, numSteps)

    # Create empty arrays to store information
    allTimes = []
    allVarsL = []
    allVarsR = []

    # Loop through until we reach the max N
    for numWalkers in numWalkersList:
        # Find max time based on N
        maxTime = int(np.log(numWalkers)**2)

        # Create arrays to track each edge
        leftEdge = []
        rightEdge = []
        # Repeat numRuns times
        for x in range(numRuns):
            # Create array of edges
            edges = bc.floatRunFixedTime(maxTime,bc.einsteinBias,numWalkers)
            # Separate arrays for left and right edges
            leftEdge.append(edges[:,0])
            rightEdge.append(edges[:,1])

        # Calculate variance of left and right edges
        varLeft = np.var(np.stack(leftEdge), axis=0)
        varRight = np.var(np.stack(rightEdge), axis=0)

        # Add current time array to array of all times (not confusing at all)
        allTimes.append(np.arange(1,maxTime+1,1))

        # Add variances for this N to array of variances
        allVarsL.append(varLeft)
        allVarsR.append(varRight)

    return allVarsL, allVarsR, allTimes, numWalkersList
