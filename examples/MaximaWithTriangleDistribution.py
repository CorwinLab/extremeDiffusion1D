#from pyDiffusion.pydiffusionPDF import DiffusionPDF
from pyDiffusion import DiffusionPDF
from pyDiffusion import pymultijumpRW as pymjRW
import numpy as np
# import npquad
from matplotlib import pyplot as plt

def RightTriangleDist(NumOfParticles,OccSize):
    """
    Generates a DiffusionPDF with underlying distribution as a Right Triangle
    distribution (a=c=1/4,b=1), with specified system of particles and occupancy size
    """
    test = DiffusionPDF(
        NumOfParticles,
        'triangular',
        [1/4, 1/4, 1],  #[a=1/4, c = a, b=1]
        OccSize
    )
    return test
def PlotTriangleHistogram(pdf):
    """
    Generates a bunch of random variables according to specified distribution and plots as histogram
    """
    vals = [pdf.generateRandomVariable() for i in range(0,1000000)]
    plt.ion()
    plt.figure()
    plt.hist(vals, bins=100,density = True)
    plt.title('Triangular Distribution Histogram')
    plt.suptitle(pdf.parameters)
    plt.show()
def MaxEdgePos(pdf,maxTime,Plot=False):
    """
    Returns array of max positions to be plotted vs time?
    Parameters:
        pdf: a pydiffusionPDF DiffusionPDF object
        maxTime: number of timesteps to move occupancy forward; <= pdf occsize, but not >
        Plot: Boolean (automatically False); if True, plots max pos. vs time on log log scale
    """
    maxEdgetest = np.zeros(maxTime) # returns a new array of given shape filled w/ zeroes
    for t in range(maxTime):
        pdf.iterateTimestep()
        #pdf.edges[1] because .edges is an array with min/max edge indicies, so 1 is the maxi
        maxEdgetest[t] = 2 * pdf.edges[1] - pdf.currentTime #fill maxEdge array with positions
    if Plot:
        fig, ax = plt.subplots()
        plt.title('Triangle Distribution: Max Particle Position vs Time')
        ax.set_xlabel("Time")
        ax.set_ylabel("Position")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim([1, maxTime])
        ax.plot(pdf.time[1:], maxEdgetest)
        ax.plot(pdf.time[1:], np.sqrt(2*pdf.time[1:]*np.log(pdf.nParticles)))
        plt.show()
    # Should I make it also return pdf.time[1:]?... I think it's fine b/c if you've already
    # defined the PDF arg. then it should have pdf.time[1:]
    return maxEdgetest
def RightTriQuantiles(NumOfParticles,occsize,maxTime,Plot=False):
    """
    Returns array of the 1/N quantiles to be plotted vs time
    Parameters:
        NumOfParticles: number of particles in system
        occsize: system size
        maxTime: number of timesteps to move occupancy forward; <= pdf occsize, but not >
        Plot: Boolean (automatically False); if True, plots quantile vs time on log log scale
    """
    pdf = np.zeros(occsize)
    pdf[0] = 1
    t = 1
    quantiles = []
    time = np.array([i for i in range(maxTime)])
    for _ in range(maxTime):  # range is number of timesteps
        pdf = pymjRW.iterateTimeStep(pdf, t, 3, 'righttriangle')
        quantiles.append(pymjRW.measureQuantile(pdf, NumOfParticles, t, 3))
        mean, var, pdf_sum = pymjRW.getMeanVarMax(pdf, NumOfParticles, t, 3)
        t += 1
    if Plot:
        fig, ax = plt.subplots()
        plt.title('Right Triangle Dist. Quantiles vs Time')
        ax.set_xlabel("Time")
        ax.set_ylabel("Quantiles")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.plot(time, quantiles)  # plot time & quantiles
        ax.plot(time, np.sqrt(2 * time * np.log(NumOfParticles))) #plot time & SSRW expectation
        plt.show()
    return quantiles

