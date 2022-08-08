import sys
import os
import numpy as np
import npquad
from typing import Iterable, Tuple, List

from .fileIO import saveArrayQuad
from .lDiffusionLink import libDiffusion


class FirstPassagePDF(libDiffusion.FirstPassagePDF):
    """Object to simulate the probability distribution of the
    first passage time.

    Parameters
    ----------
    beta : float
        Value of beta to draw random numbers from. Note that
        beta = inf is the deterministic case where all the
        transition probabilities are 0.5.

    maxPosition : int
        Maximum position to run the system out to.

    staticEnvironment : bool (False)
        Whether or not to keep the environment, or transition probabilities,
        constant in time.
    """

    def __init__(self, beta: float, maxPosition: int, staticEnvironment: bool = False):
        super().__init__(beta, maxPosition, staticEnvironment)

    @property
    def currentTime(self):
        return self.getTime()

    @currentTime.setter
    def currentTime(self, time):
        self.setTime(time)

    @property
    def beta(self):
        return self.getBeta()

    @beta.setter
    def beta(self, beta):
        self.setBeta(beta)

    @property
    def pdf(self):
        return self.getPDF()

    @pdf.setter
    def pdf(self, pdf):
        self.setPDF(pdf)

    @property
    def maxPosition(self):
        return self.getMaxPosition()

    @maxPosition.setter
    def maxPosition(self, maxPosition):
        self.setMaxPosition(maxPosition)

    @property
    def firstPassageCDF(self):
        return self.getFirstPassageCDF

    def iterateTimeStep(self):
        super().iterateTimeStep()

    def evolveToTime(self, time: int):
        while self.currentTime < time:
            self.iterateTimeStep()

    def evolveAndSaveFirstPassageCDF(self, times: Iterable[int], file: str):
        """Evolve and save the first passage time pdf

        Parameters
        ----------
        times : list or np array
            Times to save the first passage time pdf for
        file : str
            File to store data at

        Example
        -------
        >>> from matplotlib import pyplot as plt
        >>> beta = np.inf
        >>> maxPosition = 50
        >>> file = 'Data.txt'
        >>> times = np.arange(1, 10000)
        >>> pdf = FirstPassagePDF(beta, maxPosition)
        >>> pdf.evolveAndSaveFirstPassageCDF(times, file)
        >>> data = np.loadtxt(file)
        >>> fig, ax = plt.subplots()
        >>> ax.plot(data[:, 0][1::2], data[:, 1][1::2])
        >>> fig.savefig("CDF.png")
        """
        cdf = np.zeros(len(times), dtype=np.quad)
        for i, t in enumerate(times):
            self.evolveToTime(t)
            cdf[i] = self.firstPassageCDF
        saveArrayQuad(file, np.array([times, cdf]).T)

    def evolveToCutoff(self, cutoff: float, nParticles: np.quad) -> Tuple[int, np.quad]:
        """Evolve the system until it reaches a threshold for the
        cumulative distribution function of the Nth particle.

        Parameters
        ----------
        cutoff : float or np.quad
            Probability threshold that determines when to stop the
            simulation. When the CDF of the Nth particle reaches this
            value the function stops. We also need the 1/N quantile
            position so will continue until the cdf of a single particle
            is at least 1/N.

        nParticles : np.quad or float
            Number of particles

        Returns
        -------
        tuple (int, np.quad)
            First value is the N-quantile of first passage time for a single particle.
            The second value is the variance of the first passage time for a N-particle
            system.

        Examples
        --------
        >>> maxPosition = 500
        >>> beta = np.inf
        >>> pdf = FirstPassagePDF(beta, maxPosition)
        >>> data = pdf.evolveToCutoff(1, 10**24)
        >>> print(data)
        (2360 2796.11819544274829798130996166195575)
        """
        return super().evolveToCutoff(cutoff, nParticles)

    def evolveToCutoffMultiple(
        self, nParticles: Iterable[np.quad], cutoff: float = 1.0
    ) -> Tuple[List[int], List[np.quad], List[np.quad]]:
        """
        Evolve the system until it reaches a threshold for the cumalitive distribution function
        of all specified N particles.

        Parameters
        ----------
        nParticles : list or array
            List of number of particles to measure first passage time for

        cutoff : float (1.0)
            Probability threshold that determines when to stop the
            simulation. This should always be 1.0 or the variance won't be
            reliable. When the CDF of the Nth particle reaches this
            value the function stops. We also need the 1/N quantile
            position so will continue until the cdf of a single particle
            is at least 1/N.

        Returns
        -------
        tuple (list(int), list(np.quad), list(np.quad))
            First entry is time of 1/N quantile. Second is variance of N particle
            distribution. Third is order of nParticles. Note, the reason the nParticles
            are returned is because the algorithm may record the quantile and variance
            out of order compared to the input.

        Examples
        --------
        >>> from pyDiffusion import FirstPassagePDF
        >>> import numpy as np
        >>> import npquad 
        >>> maxPosition = 500
        >>> beta = np.inf 
        >>> nParticles = [1e10, 1e24, 1e50]
        >>> pdf = FirstPassagePDF(beta, maxPosition)
        >>> data = pdf.evolveToCutoffMultiple(cutoff=1, nParticles=nParticles)
        >>> print(data)
        ([1146, 2360, 5798], [139.750323806665726286823820804983635, 2796.11819544274830140495584666294565, 92820.89093004149373964542503490102], [1.00000000000000007629769841091887003e+50, 999999999999999983222784, 10000000000])
        """

        return super().evolveToCutoffMultiple(cutoff, nParticles)