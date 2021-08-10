import recurrance
import numpy as np
import npquad
import csv

class Recurrance(recurrance.Recurrance):
    """
    Create a class that models the recurrance relation outlined in the BC
    model paper.

    Attributes
    ----------
    zB : numpy array (dtype of np.quad)
        The current recurrance vector zB(n, t)

    time : int
        Current time in the recurrance relation

    beta : float
        Value of beta used in the recurrance relation
    """

    def __str__(self):
        return f"Recurrance(beta={self.beta}, time={self.time})"

    def __repr__(self):
        return self.__str__()

    @property
    def beta(self):
        return self.getBeta()

    @property
    def zB(self):
        return np.array(self.getzB(), dtype=np.quad)

    @property
    def time(self):
        return self.getTime()

    @property
    def tMax(self):
        return self.gettMax()

    def iterateTimeStep(self):
        """
        Evolve the recurrance relation zB forward one step in time.

        Raises
        ------
        ValueError
            If trying to evolve the system to a time greater than the allocated
            time. This would normally Core Dump since trying to allocate memory
            outside array. 
        """

        if self.time >= self.tMax:
            raise ValueError("Cannot evolve to time greater than tmax")

        super().iterateTimeStep()

    def evolveToTime(self, time):
        """
        Evolve the system to a time t.

        Parameters
        ----------
        time : int
            Time to iterate the system forward to
        """

        while self.time < time:
            self.iterateTimeStep()

    def evolveTimesteps(self, num):
        """
        Evolve the system forward a number of timesteps.

        Parameters
        ----------
        num : int
            Number of timesteps to evolve the system
        """

        for _ in range(num):
            self.iterateTimeStep()

    def findQuintile(self, N):
        """
        Find the corresponding quintile.

        Parameters
        ----------
        N : np.quad
            Nth quartile to measure

        Returns
        -------
        int
            Position of Nth quartile
        """

        return super().findQuintile(N)

    def findQuintiles(self, Ns, descending=False):
        """
        Find the corresponding quintiles. Should be faster than a list compression
        over findQuntile b/c it does it in one loop.

        Parameters
        ----------
        Ns : numpy array (dtype np.quad)
            Nth quartiles to measure

        descending : bool
            Whether or not the incoming Ns are in descending or ascending order.
            If they are not in descending order we flip the output quintiles.

        Returns
        -------
        numpy array (dtype ints)
            Position of Nth quartiles
        """

        if descending:
            return np.array(super().findQuintiles(Ns))
        else:
            returnVals = super().findQuintiles(Ns)
            returnVals.reverse()
            return np.array(returnVals)

    def evolveAndSaveQuartile(self, time, quartiles, file):
        """
        Evolve the system to specific times and save the quartiles at those times
        to a file.

        Parameters
        ----------
        time : numpy array or list
            Times to evolve the system to and save quartiles at

        quartiles : numpy array (dtype np.quad)
            Quartiles to save at each time

        file : str
            File to save the quartiles to.

        Examples
        --------
        >>> r = Recurrance(beta=np.inf)
        >>> r.evolveAndSaveQuartile([1, 5, 50], [5, 10], 'Data.txt')
        """

        f = open(file, "w")
        writer = csv.writer(f)
        header = ["time"] + [str(q) for q in quartiles]
        writer.writerow(header)
        for t in time:
            self.evolveToTime(t)

            quartiles = list(np.array(quartiles))
            quartiles.sort() # Need to get the quartiles in descending order
            quartiles.reverse()
            NthQuintiles = self.findQuintiles(quartiles)

            row = [self.time] + list(NthQuintiles)
            writer.writerow(row)
        f.close()
