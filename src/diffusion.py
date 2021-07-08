import numpy as np
import sys
sys.path.append('../cDiffusion')
import cDiffusion as cdiff
from datetime import datetime
import os
import csv

class Diffusion(cdiff.Diffusion):
    '''
    Helper class for C++ Diffusion object.
    '''

    def __str__(self):
        return f"Diffusion(N={self.getN()}, beta={self.getBeta()})"

    def __repr__(self):
        return self.__str__()

    @property
    def center(self):
        return np.arange(self.getTime()) * 0.5

    @property
    def minDistance(self):
        minEdge = self.getEdges()[0]
        return minEdge - self.center

    @property
    def maxDistance(self):
        maxEdge = self.getEdges()[1]
        return maxEdge - self.center

    def evolveSaveOccupancy(self, times):
        '''
        Evolve the system forward in "chuncks" of time and save the occupancy after
        each chunk of time.

        Parameters
        ----------
        times : list
            Time steps to save the occupancy at.

        Returns
        -------
        occupancies : list
            Occupancy at each time
        '''
        occupancies = []
        dt = np.diff(times)
        for t in dt:
            print(t)
            self.evolveTimesteps(t, inplace=True)
            occ = self.getOccupancy()
            occupancies.append(occ)

        return occupancies

    def saveEdges(self, filename=None):
        '''
        Saves the edges to a txt file. If no filename is provided defaults
        to saving as the date and time. Also stores metadata about the simulation
        such as N, beta, smallCutoff, largeCutoff to a csv called "metadata.csv".

        Parameters
        ----------
        filename : str
            Edges save filename

        Examples
        --------
        >>> d = Diffusion(50, 1)
        >>> occ = np.zeros(50)
        >>> occ[0] = 50
        >>> d.setOccupancy(occ)
        >>> d.evolveTimesteps(10)
        >>> edges = d.getEdges()
        >>> d.saveEdges('./myEdges.txt')
        '''

        edges = np.array(self.getEdges()).T
        if filename is None:
            filename = datetime.now().strftime('%y-%m-%d--%H-%M-%S.txt')
        vars = {'filename': os.path.basename(filename), 'N': self.getN(),
                'beta': self.getBeta(), 'smallCutoff': self.getsmallCutoff(),
                'largeCutoff': self.getlargeCutoff()}

        meta_filename = 'metadata.csv'
        filepath = os.path.abspath(filename)
        dir = os.path.dirname(filepath)
        meta_filename = os.path.join(dir, meta_filename)
        if os.path.isfile(meta_filename):
            with open(meta_filename, 'a') as file:
                writer = csv.DictWriter(file, fieldnames=vars.keys())
                writer.writerow(vars)
        else:
            with open(meta_filename, 'w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=vars.keys())
                writer.writeheader()
                writer.writerow(vars)

        np.savetxt(filename, edges)

if __name__ == '__main__':
    N = 100
    times = np.geomspace(1, round(np.log(N) ** (5/2)), 1000, dtype=np.int64)
    times = np.unique(times)
    d = Diffusion(N, 1.0)
    d.initializeOccupationAndEdges()
    occs = d.evolveSaveOccupancy(times)
