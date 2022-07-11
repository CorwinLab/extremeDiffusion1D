import sys
import os
import numpy as np
import npquad
from fileIO import saveArrayQuad, loadArrayQuad

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "FirstPassagePDF")
sys.path.append(path)

import firstPassagePDF


class FirstPassagePDF(firstPassagePDF.FirstPassagePDF):
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
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
    def firstPassageProbability(self):
        return self.getFirstPassageProbability()

    def iterateTimeStep(self):
        super().iterateTimeStep()

    def evolveToTime(self, time):
        while self.currentTime < time:
            self.iterateTimeStep()

    def evolveAndSaveFirstPassagePDF(self, times, file):
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
        >>> pdf.evolveAndSaveFirstPassagePDF(times, file)
        >>> data = np.loadtxt(file)
        >>> fig, ax = plt.subplots()
        >>> ax.plot(data[:, 0][1::2], data[:, 1][1::2])
        >>> fig.savefig("PDF.png")
        """
        pdf = np.zeros(len(times), dtype=np.quad)
        for i, t in enumerate(times):
            self.evolveToTime(t)
            pdf[i] = self.firstPassageProbability
        saveArrayQuad(file, np.array([times, pdf]).T)

    def evolveToCutoff(self, cutoff):
        return np.array(super().evolveToCutoff(cutoff)).T
