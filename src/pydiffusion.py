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
        '''
        Returns center of the occupancy over time.
        '''

        return np.arange(0, self.getTime()+1) * 0.5

    @property
    def minDistance(self):
        '''
        Returns the distance from the left side of the occupancy over time.
        '''

        minEdge = self.getEdges()[0]
        return minEdge - self.center

    @property
    def maxDistance(self):
        '''
        Returns the distance from the right side of the occupancy over time. This
        is generally the one we care about.
        '''

        maxEdge = self.getEdges()[1]
        return maxEdge - self.center

    def evolveTimeSteps(self, iterations):
        '''
        Evolves the system forward a specified number of timesteps.

        Parameters
        ----------
        iterations : int
            Number of timesteps to iterate forward
        '''

        for _ in range(iterations):
            self.iterateTimestep()

    def evolveToTime(self, time):
        '''
        Evolve the system to a specified time. If the input time is greater than
        the system's current time it won't actually do anything.

        Parameters
        ----------
        time : int
            System time to evolve the system forward to
        '''

        while (self.getTime() < time):
            self.iterateTimestep()

    def evolveAndSaveQuartile(self, time, quartiles, file):
        '''
        Incrementally evolves the system forward to the specified times and saves
        the specified quartiles after each increment.

        Parameters
        ----------
        time : list or numpy array
            Times to evolve the system to and save the quartiles at

        quartiles : list or numpy array
            Quartiles to save at each time. These should all be < 1.

        file : string
            Filename (or path) to save the time and quartiles

        Examples
        --------
        >>> N = 1e300
        >>> num_of_steps = round(3 * np.log(N) ** (5/2))
        >>> d = Diffusion(N, beta=beta, occupancySize=num_of_steps, smallCutoff=0, largeCutoff=0, probDistFlag=True)
        >>> save_times = np.geomspace(1, num_of_steps, 1000, dtype=np.int64)
        >>> save_times = np.unique(save_times)
        >>> quartiles = [10 ** i for i in range(20, 280, 20)]
        >>> quartiles = [1/i for i in quartiles]
        >>> save_file = 'Data.txt'
        >>> d.evolveAndSaveQuartile(save_times, quartiles, save_file)

        Notes
        -----
        Looks like this is a bit faster than the evolveAndSave method which
        stores everything to a numpy array and then saves it.
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
        '''
        Incrementally evolves the system forward to the specified times and saves
        the number of particles greater than position v * time after each increment.
        This is to evaluate Pb(vt, t) at the specified times where Pb(x, t) is the
        probability of a particle being greater than x at time t. Need to divide
        by nParticles to get the probability.

        Parameters
        ----------
        time : list or numpy array
            Times to evolve the system to and save the quartiles at

        vs : list or numpy array
            Velocities to save at each time. These should all satisfy 0 <= v <= 1.
            Note that v=1 corresponds to the probability of a particle being greater
            than x=t and v=0 corresponds to the probability of a particle being
            greater than x=t/2 or the center.

        file : string
            Filename (or path) to save the time and probabilities

        Examples
        --------
        >>> N = 1e300
        >>> num_of_steps = 100_000
        >>> d = Diffusion(N, beta=beta, occupancySize=num_of_steps, smallCutoff=0, largeCutoff=0, probDistFlag=True)
        >>> save_times = np.geomspace(1, num_of_steps, 1000, dtype=np.int64)
        >>> save_times = np.unique(save_times)
        >>> vs = np.geomspace(1e-7, 1, 50)
        >>> save_file = 'Data.txt'
        >>> d.evolveAndSaveV(save_times, vs, save_file)

        Notes
        -----
        Looks like this is a bit faster than the evolveAndSave method which
        stores everything to a numpy array and then saves it.

        To get the index we use the formula:
                        idx = round((v * t + t) / 2)
        which satisfies the constraints mentioned in the velocity definition.
        '''

        f = open(file, 'w')
        writer = csv.writer(f)
        header = ['time'] + [str(v) for v in vs]
        writer.writerow(header)
        for t in time:
            self.evolveToTime(t)
            idx = (self.getTime() * vs + self.getTime()) / 2
            idx = np.round(idx).astype(np.int64)
            pos = [self.pGreaterThanX(i) for i in idx]
            row = [self.getTime()] + pos
            writer.writerow(row)
        f.close()

    def evolveAndSave(self, time, quartiles, file):
        '''
        Incrementally evolves the system forward to the specified times and saves
        the specified quartiles after each increment. The data is stored as a
        numpy array which may make it slower than the evolveAndSaveQuartile method.

        Parameters
        ----------
        time : list or numpy array
            Times to evolve the system to and save the quartiles at

        quartiles : list or numpy array
            Quartiles to save at each time. These should all be < 1.

        file : string
            Filename (or path) to save the time and quartiles

        Notes
        -----
        Looks like this is a bit slower than the evolveAndSaveQuartile method which
        stores writes to the file incrementally.
        '''

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

        Parameters
        ----------
        vs : list or numpy array
            Velocities to print out

        timesteps : int
            How many timesteps to iterate forward (generally good to keep this small
            so you can actually add up the occupancy)

        Examples
        --------
        # Note the output will change each time this is run since the biases are random
        >>> N = 1e300
        >>> d = Diffusion(N, beta=1, occupancySize=10, smallCutoff=0, largeCutoff=0, probDistFlag=True)
        >>> d.ProbBiggerX(np.array([0.5, 1]), 1)
        Bigger than Index: [3.058954085425106e+299, 3.058954085425106e+299]
        Indices:  [1 1]
        Occupancy: [6.94104591e+299 3.05895409e+299]
        Prob:  [0.30589541 0.30589541]
        '''

        for _ in range(timesteps):
            self.iterateTimestep()

        # It looks like this produces the proper indeces we are looking for!
        idx = (self.getTime() * vs + self.getTime()) / 2
        idx = np.round(idx).astype(np.int64)

        nonzeros = np.nonzero(self.getOccupancy())[0]
        Ns = [self.pGreaterThanX(i) for i in idx]
        print('Bigger than Index:', Ns)
        print('Indices: ', idx)
        print('Occupancy:', np.array(self.getOccupancy())[nonzeros])
        print('Prob: ', np.array(Ns)/self.getNParticles())
