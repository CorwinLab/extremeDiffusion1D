import sys
import os

# Need to link to diffusionPDF library (PyBind11 code)
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DiffusionCDF")
sys.path.append(path)

import diffusionCDF
import numpy as np
import npquad
import csv


class DiffusionTimeCDF(diffusionCDF.DiffusionTimeCDF):
    """
    Create a class that iterates through the time of the CDF.

    Attributes
    ----------
    CDF : numpy array (dtype of np.quad)
        The current recurrance vector CDF(n, t)

    time : int
        Current time in the recurrance relation

    beta : float
        Value of beta used in the recurrance relation
    """

    def __str__(self):
        return f"DiffusionCDF(beta={self.beta}, time={self.time})"

    def __repr__(self):
        return self.__str__()

    @property
    def time(self):
        return self.getTime()

    @property
    def beta(self):
        return self.getBeta()

    @property
    def CDF(self):
        return self.getCDF()

    @property
    def tMax(self):
        return self.gettMax()

    @property
    def setBetaSeed(self, seed):
        super().setBetaSeed(seed)

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

    def findQuantile(self, quantile):
        """
        Find the corresponding quantile.

        Parameters
        ----------
        quantile : np.quad
            Quantile to measure

        Returns
        -------
        int
            Position of Nth quantile
        """

        return super().findQuantile(quantile)

    def findQuantiles(self, quantiles, descending=False):
        """
        Find the corresponding quantiles. Should be faster than a list compression
        over findQuntile b/c it does it in one loop.

        Parameters
        ----------
        quantiles : numpy array (dtype np.quad)
            Nth quantiles to measure

        descending : bool
            Whether or not the incoming quantiles are in descending or ascending order.
            If they are not in descending order we flip the output quantiles.

        Returns
        -------
        numpy array (dtype ints)
            Position of Nth quantiles
        """

        if descending:
            return np.array(super().findQuantiles(quantiles))
        else:
            returnVals = super().findQuantiles(quantiles)
            returnVals.reverse()
            return np.array(returnVals)

    def evolveAndSaveQuantile(self, time, quantiles, file):
        """
        Evolve the system to specific times and save the quantiles at those times
        to a file.

        Parameters
        ----------
        time : numpy array or list
            Times to evolve the system to and save quantiles at

        quantiles : numpy array (dtype np.quad)
            Quantiles to save at each time

        file : str
            File to save the quantiles to.

        Examples
        --------
        >>> r = Recurrance(beta=np.inf)
        >>> r.evolveAndSaveQuantile([1, 5, 50], [5, 10], 'Data.txt')
        """

        f = open(file, "w")
        writer = csv.writer(f)
        header = ["time"] + [str(q) for q in quantiles]
        writer.writerow(header)
        for t in time:
            self.evolveToTime(t)

            quantiles = list(np.array(quantiles))
            quantiles.sort()  # Need to get the quantiles in descending order
            quantiles.reverse()
            NthQuantiles = self.findQuantiles(quantiles)

            row = [self.time] + list(NthQuantiles)
            writer.writerow(row)
        f.close()


class DiffusionPositionCDF(diffusionCDF.DiffusionPositionCDF):
    """
    Class to iterate through the position of the CDF.
    """

    def __str__(self):
        return f"DiffusionCDF(beta={self.beta}, time={self.position})"

    def __repr__(self):
        return self.__str__()

    @property
    def beta(self):
        return self.getBeta()

    @property
    def CDF(self):
        return self.getCDF()

    @property
    def tMax(self):
        return self.gettMax()

    @property
    def setBetaSeed(self, seed):
        super().setBetaSeed(seed)

    @property
    def position(self):
        return self.getPosition()

    @property
    def quantilePositions(self):
        return self.getQuantilesMeasurement()

    @property
    def quantiles(self):
        return self.getQuantiles()

    def stepPosition(self):
        """
        Move the system forward one step in the position.
        """

        if self.position == self.tMax:
            raise ValueError("Cannot evolve the system to a position larger than tMax")

        super().stepPosition()

    def evolveToPosition(self, n):
        """
        Move the system forward to a specified position.

        Parameters
        ----------
        n : int
            Position to move the system forward to.
        """

        while self.position < n:
            self.stepPosition()

    def evolvePositions(self, num_positions):
        """
        Move the system forward a fixed number of positions.
        """

        for _ in range(num_positions):
            self.stepPosition()
