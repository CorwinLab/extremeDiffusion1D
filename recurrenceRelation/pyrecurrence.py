import recurrance
import numpy as np
import npquad

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

    def iterateTimeStep(self):
        """
        Evolve the recurrance relation zB forward one step in time.
        """

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
