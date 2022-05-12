import numpy as np
import sys 
sys.path.append("../ParallelEnvironments")
import diffusionSystems

class AllSystems(diffusionSystems.AllSystems):
    def __init__(self, numSystems, beta, xMax, numWalkers):
        """
        Initialize an environment to run random walks in.

        Parameters
        ----------
        numSystems : int
            Number of systems to evolve and average over.
        beta : float
            Value of beta to use in the beta distribtuion to generate
            random biases. Currently, if beta==inf then all biases 
            are 1/2 and if beta != inf the biases are drawn from a 
            uniform distribution.
        xMax : int
            Size of the array to build for each system.
        numWalkers : int
            Number of walkers to use in each system. Currently doesn't 
            support quad precision numbers.
        """
        super().__init__(numSystems, beta, xMax, numWalkers)

    def __str__(self): 
        return f"AllSystems(numSystems={self.numSystems}, beta={self.beta}, xMax={self.xMax}, numWalkers={self.numWalkers})"

    def __repr__(self):
        return self.__str__()

    def __len__(self): 
        return self.getNumSystems()

    def getAllOccupancies(self):
        """
        Get all the occupancies in the system. Mainly used for debugging 
        to see what the systems look like.

        Returns
        -------
        occupancy : numpy array
            Number of particles at each site in the system. Size is 
            (numSystems, xMax).
        """
        occupancy = np.zeros([len(self), self.getXMax()])
        for i, Sys in enumerate(self.getSystems()): 
            occupancy[i, :] = np.array(Sys.occupancy)
        return occupancy

    def getEdges(self):
        """
        Returns the minimum and maximum particle indeces for each system. 

        Returns
        -------
        edges : numpy array 
            Min/Max index containing a particle for every system. Size is 
            (numSystems, 2) where the columns are (min, max).
        """
        edges = np.zeros([len(self), 2])
        for i, Sys in enumerate(self.getSystems()): 
            edges[i, :] = [Sys.getMinPos(), Sys.getMaxPos()]
        return edges

    def iterateTimeStep(self, sysIDs=None):
        """
        Move specified systems forward in time. 

        Parameters
        ----------
        sysIDs : list[int], optional
            System IDs to move forward in time, by default None. If None
            provided movels all systems forward. The system ID is the 
            index in the systems list.
        """
        if sysIDs is None: 
            sysIDs = list(range(0, len(self)))
        
        super().iterateTimeStep(sysIDs)

    def evolveToTime(self, time):
        """
        Evolve all systems forward to a specified time.

        Parameters
        ----------
        time : int
            Time to evolve all the systems forward to.
        """
        for _ in range(time): 
            self.iterateTimeStep()

    def measureFirstPassageTimes(self, distances):
        """
        Measure the first passage time

        Parameters
        ----------
        distances : list[int]
            Distances in space to measure first passage time of. Note
             these are distict from the indeces of the occupancy array.

        Returns
        -------
        numpy array
            First passage times for each system. Array size is (numSystems, len(distances))
        """
        return np.array(super().measureFirstPassageTimes(distances))

if __name__ == '__main__':
    import time
    tMax = 1000
    numSystems = 100
    nParticles=1e10
    system = AllSystems(numSystems, 1, tMax, nParticles)

    s = time.time()
    for t in range(tMax-1):
        system.iterateTimeStep()
    print(time.time() - s)

    system = AllSystems(10, float('inf'), 10, 1000)
    system.iterateTimeStep()
    system.iterateTimeStep()
    print(system.getAllOccupancies())

    system = AllSystems(10, float('inf'), 10, 10)
    firstTimes = system.measureFirstPassageTimes([1, 2, 3, 4])
    print(np.array(firstTimes))