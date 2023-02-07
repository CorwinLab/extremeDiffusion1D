import numpy as np 
import npquad 
from .lDiffusionLink import libDiffusion
import csv

class DiffusionND(libDiffusion.DiffusionND):
    def __init__(self, alpha, tMax, L):
        '''
        from matplotlib import pyplot as plt
        from matplotlib import colors
        tMax = 510
        L = 2
        d = DiffusionND(4*[1], tMax, L)
        for t in range(tMaxs
            d.iterateTimestep()
        cmap = plt.get_cmap("cool")  # Can be any colormap that you want after the cm
        cmap.set_bad(color='white')

        cdf = d.CDF.astype(float)
        cdf[cdf==0] = -1
        fig, ax = plt.subplots()
        img = ax.imshow(cdf, cmap=cmap, norm=colors.LogNorm(10**-15, 1))

        cbar = fig.colorbar(img, ax=ax)
        cbar.ax.set_ylabel("Probability Density")
        fig.savefig(f"CDF.png", bbox_inches='tight')
        plt.close(fig)
        '''
        super().__init__(alpha, tMax, L)

    @property 
    def CDF(self):
        return np.array(self.getCDF())

    @CDF.setter 
    def CDF(self, _CDF):
        self.setCDF(_CDF)

    def iterateTimestep(self):
        if (self.time+1) >= self.tMax:
            raise RuntimeError(f"At time {self.time} but cannot iterate past time {self.tMax}")
        
        super().iterateTimestep()

    @property 
    def tMax(self):
        return self.gettMax()

    @property 
    def time(self):
        return self.getTime()
    
    @property
    def L(self):
        return self.getL()
    
    @property 
    def absorbedProb(self):
        return self.getAbsorbedProb()
    
    def getQuantileAndVariance(self, N):

        quantile = None
        running_sum_squared = 0
        running_sum = 0
        firstPassageCDF = self.absorbedProb
        nFirstPassageCDFPrev = 1 - np.exp(-N * firstPassageCDF)

        while (nFirstPassageCDFPrev < 1) or (firstPassageCDF < 1/N):
            self.iterateTimestep()
            #print(self.time)

            firstPassageCDF = self.absorbedProb
            nFirstPassageCDF = 1 - np.exp(-N * firstPassageCDF)
            
            nFirstPassagePDF = nFirstPassageCDF - nFirstPassageCDFPrev
            running_sum_squared += self.time ** 2 * nFirstPassagePDF
            running_sum += self.time * nFirstPassagePDF

            if (quantile is None) and (firstPassageCDF > 1/N):
                quantile = self.time

            nFirstPassageCDFPrev = nFirstPassageCDF
        
        variance = running_sum_squared - running_sum ** 2
        return quantile, variance
    