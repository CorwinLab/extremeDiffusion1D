import numpy as np
import sys
import os
sys.path.append(os.path.abspath('../cDiffusion'))
import diffusion as cdiff
import csv
import time

class Diffusion(cdiff.Diffusion):
    '''
    Helper class for C++ Diffusion object.
    '''

    def __str__(self):
        return f"Diffusion(N={self.getNParticles()}, beta={self.getBeta()}, size={len(self.getEdges()[0])}, time={self.getTime()})"

    def __repr__(self):
        return self.__str__()

    @property
    def center(self):
        return np.arange(0, self.getTime()+1) * 0.5

    @property
    def minDistance(self):
        minEdge = self.getEdges()[0]
        return minEdge - self.center

    @property
    def maxDistance(self):
        maxEdge = self.getEdges()[1]
        return maxEdge - self.center

    def evolveTimeSteps(self, iterations):
        for _ in range(iterations):
            self.iterateTimestep()

    def evolveToTime(self, time):
        while (self.getTime() < time):
            self.iterateTimestep()

    def evolveAndSaveQuartile(self, time, quartiles, file):
        '''
        Looks like this is a bit faster than the evolveAndSave method which
        saves everything to a numpy array and then saves it.
        '''
        f = open(file, 'w')
        writer = csv.writer(f)
        header = ['time', 'MaxEdge'] + ['{:.0e}'.format(1/i) for i in quartiles]
        writer.writerow(header)
        for t in time:
            self.evolveToTime(t)
            NthQuartile = [self.NthquartileSingleSided(self.getNParticles() * q) for q in quartiles]
            maxEdge = self.getEdges()[1][t]
            row = [self.getTime(), maxEdge] + NthQuartile
            writer.writerow(row)
        f.close()

    def evolveAndSaveV(self, time, vs, file):
        f = open(file, 'w')
        writer = csv.writer(f)
        header = ['time'] + [str(v) for v in vs]
        writer.writerow(header)
        for t in time:
            self.evolveToTime(t)
            idx = (self.getTime() * vs + self.getTime()) / 2
            idx = np.round(idx)
            pos = [self.pGreaterThanX(i) for i in idx]
            row = [self.getTime()] + pos
            writer.writerow(row)
        f.close()

    def evolveAndSave(self, time, quartiles, file):
        save_array = np.zeros(shape=(len(time), len(quartiles)+2))
        for row_num, t in enumerate(time):
            self.evolveToTime(t)
            NthQuartile = [self.NthquartileSingleSided(self.getNParticles() * q) for q in quartiles]
            maxEdge = self.getEdges()[1][t]
            row = [self.getTime(), maxEdge] + NthQuartile
            save_array[row_num, :] = row
        np.savetxt(file, save_array)

    def ProbBiggerX(self, vs, timesteps):
        '''
        Troubleshooting function to make sure that pGreaterThanX function
        works properly.

        Examples
        --------
        >>> N = 1e300
        >>> d = Diffusion(N, beta=1, occupancySize=10, smallCutoff=0, largeCutoff=0, probDistFlag=True)
        >>> d.ProbBiggerX(np.array([0.5, 1]), 1)
        '''
        for _ in range(timesteps):
            self.iterateTimestep()

        # it looks like this produces the proper indeces we are looking for!
        idx = (self.getTime() * vs + self.getTime()) / 2
        idx = np.round(idx).astype(np.int64)

        nonzeros = np.nonzero(self.getOccupancy())[0]
        Ns = [self.pGreaterThanX(i) for i in idx]
        print('Bigger than Index:', Ns)
        print('Indices: ', idx)
        print('Occupancy:', np.array(self.getOccupancy())[nonzeros])
        print('Prob: ', np.array(Ns)/1e300)
