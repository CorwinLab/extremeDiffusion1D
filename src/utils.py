# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 16:09:48 2021

@author: jacob
"""

from joblib import Parallel, delayed
import multiprocessing
import os
from BarraquandCorwin import floatRunFixedTime, einsteinBias
import numpy as np

def parallelVarianceMean(maxTime, biasFunction, numSamples, numWalkers=None):
    ''' Compute the variance and mean as a function of time for a fixed number of samples

    args:
        maxTime (int): The maximum number of timesteps
        biasFunction (function): A function that computes the bias for each site
        numWalkers (float): The number of walkers
        numSamples (int): The number of systems to create
    '''
    numCores = multiprocessing.cpu_count()//2
    with Parallel( n_jobs = numCores ) as parallel:
        edges = parallel(delayed(floatRunFixedTime)(maxTime, biasFunction, numWalkers=numWalkers) for i in range(numSamples))
        # results = parallel(delayed(runFixedTime)(tMax, biasFunction) for i in range(numSamples))
    edges = np.stack(edges)
    return range(1, maxTime+1), np.mean(edges, axis=0), np.var(edges, axis=0)
    # return edges
    # return np.hstack(edges)

def computeVarianceMean(topDir, N, beta):
    edges = []
    for fileName in os.listdir(topDir):
        tempEdge = np.max(np.abs(np.loadtxt(topDir + os.sep + fileName)),1)
        edges.append( tempEdge )
        print(fileName)
    edges = np.stack(edges)
    maxTime = edges[0].shape[0]
    return range(1, maxTime+1), np.mean(edges, axis=0), np.var(edges, axis=0), N, beta

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
            edges = floatRunFixedTime(maxTime,einsteinBias,numWalkers)
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