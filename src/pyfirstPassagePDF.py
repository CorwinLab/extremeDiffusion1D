import sys
import os 
import numpy as np 
import npquad
from fileIO import saveArrayQuad, loadArrayQuad

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "FirstPassagePDF")
sys.path.append(path)

import firstPassagePDF

class FirstPassagePDF(firstPassagePDF.FirstPassagePDF):
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

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from quadMath import prettifyQuad
    beta = np.inf
    maxPosition = 5000
    file = 'Data.txt'
    N = np.quad('1e24')
    logN = np.log(N).astype(float)
    times = np.arange(1, 100000)
    pdf = FirstPassagePDF(beta, maxPosition)
    pdf.evolveAndSaveFirstPassagePDF(times, file)
    data = loadArrayQuad(file)
    pdf_distribution = data[:, 1][1::2]
    times = data[:, 0][1::2].astype(float)
    cdf_distribution = np.cumsum(pdf_distribution)
    N_particle_cdf = 1-np.exp(-cdf_distribution*N)
    N_particle_pdf = np.diff(N_particle_cdf)
    fig, ax = plt.subplots(ncols=2, nrows=2, sharex=True)
    fig.suptitle(f"First Passage Probabilities for Distance={maxPosition}", fontsize=16)
    ax[0][0].plot(times, pdf_distribution)
    ax[0][0].set_ylabel("PDF")
    ax[0][0].set_title("Single Particle")
    ax[1][0].plot(times, cdf_distribution)
    ax[1][0].set_ylabel("CDF")
    ax[1][0].set_xlabel("Time")
    ax[0][1].set_title(f"N={prettifyQuad(N)}")
    ax[1][1].plot(times, N_particle_cdf)
    ax[1][1].set_xscale("log")
    ax[0][1].plot(times[1:], N_particle_pdf)
    ax[0][1].set_yscale("log")
    plt.tight_layout()
    fig.savefig("PDF.png", bbox_inches='tight')