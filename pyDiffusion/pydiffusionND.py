import numpy as np 
import npquad 
from .lDiffusionLink import libDiffusion
import csv
from numba import njit

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

def iteratePDF(pdf):
    absorbed_prob=0
    pdf_new = np.zeros(pdf.shape)
    biases = np.ones(shape=4) / 4
    for i in range(1, pdf.shape[0]-1): # rows
        for j in range(1, pdf.shape[1] - 1): # columns
            pdf_new[i+1][j] += pdf[i][j] * biases[0]
            pdf_new[i][j+1] += pdf[i][j] * biases[1]
            pdf_new[i-1][j] += pdf[i][j] * biases[2]
            pdf_new[i][j-1] += pdf[i][j] * biases[3]

    for i in range(1, pdf.shape[0]-1):
        # Left-most edge of box
        pdf_new[i+1][0] += pdf[i][0] * biases[0]
        pdf_new[i-1][0] += pdf[i][0] * biases[1]
        pdf_new[i][1] += pdf[i][0] * biases[2]
        absorbed_prob += pdf[i][0] * biases[3]

        # Right-most edge of box
        pdf_new[i+1][-1] += pdf[i][-1] * biases[0]
        pdf_new[i-1][-1] += pdf[i][-1] * biases[1]
        pdf_new[i][-2] += pdf[i][-1] * biases[2]
        absorbed_prob += pdf[i][-1] * biases[3]

    for j in range(1, pdf.shape[1]-1):
        # Top edge of box
        pdf_new[0][j-1] += pdf[0][j] * biases[0]
        pdf_new[0][j+1] += pdf[0][j] * biases[1]
        pdf_new[1][j] += pdf[0][j] * biases[2]
        absorbed_prob += pdf[0][j] * biases[3]

        # Bottom edge of box
        pdf_new[-1][j-1] += pdf[-1][j] * biases[0]
        pdf_new[-1][j+1] += pdf[-1][j] * biases[1]
        pdf_new[-2][j] += pdf[-1][j] * biases[2]
        absorbed = pdf[-1][j] * biases[3]

    # Top left corner
    absorbed_prob += pdf[0][0] * (biases[0] + biases[1])
    pdf[1][0] += pdf[0][0] * biases[2]
    pdf[0, 1] += pdf[0][0] * biases[3]
    
    
    return pdf_new, absorbed_prob

@njit
def numbaDirchlet(alpha):
    if np.isinf(alpha[0]):
        return np.ones(shape=len(alpha))/4
    else:
        gamma_random_vals = np.random.gamma(1, 1, size=len(alpha))
        return gamma_random_vals/np.sum(gamma_random_vals)

@njit
def iteratePDFSpherical(pdf, R, alpha):
    absorbed_prob=0
    pdf_new = np.zeros(pdf.shape)

    center = (pdf.shape[0]//2, pdf.shape[1] //2)
    for i in range(1, pdf.shape[0]-1): # rows
        for j in range(1, pdf.shape[1] - 1): # columns
            biases = numbaDirchlet(alpha)
            if np.sqrt((i+1-center[0])**2 + (j - center[1])**2) >= R:
                absorbed_prob += pdf[i][j] * biases[0]
            else:
                pdf_new[i+1][j] += pdf[i][j] * biases[0]

            if np.sqrt((i-center[0])**2 + (j+1-center[1])**2) >= R:
                absorbed_prob+= pdf[i][j] * biases[1]
            else:
                pdf_new[i][j+1] += pdf[i][j] * biases[1]
            
            if np.sqrt((i-1-center[0])**2 + (j-center[1])**2) >= R:
                absorbed_prob += pdf[i][j] * biases[2]
            else:
                pdf_new[i-1][j] += pdf[i][j] * biases[2]

            if np.sqrt((i-center[0])**2 + (j-1-center[1])**2) >= R:
                absorbed_prob += pdf[i][j] * biases[3]
            else:
                pdf_new[i][j-1] += pdf[i][j] * biases[3]
    
    return pdf_new, absorbed_prob

def initiateSphericalPDF(R):
    size = 2 * R
    pdf = np.zeros((size+1, size+1))
    pdf[pdf.shape[0]//2, pdf.shape[1]//2] = 1
    return pdf
