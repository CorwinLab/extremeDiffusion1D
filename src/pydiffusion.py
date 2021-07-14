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
            idx = self.getTime() * vs + self.getTime() * 0.5
            pos = [self.pGreaterThanX(int(i)) for i in idx]
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
