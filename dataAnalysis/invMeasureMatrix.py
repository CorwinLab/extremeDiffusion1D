import numpy as np
from matplotlib import pyplot as plt
from symmetricDirichlet import getSecondMomentArraySymmetricArbitraryAlpha
from scipy.sparse import linalg

if __name__ == '__main__':
    def getInvMeasure(alphas, size):
        firstMoments, secondMoments = getSecondMomentArraySymmetricArbitraryAlpha(alphas)
        
        transitionMatrix = np.zeros([size, size])
        k = transitionMatrix.shape[0] // 2
        for i in range(transitionMatrix.shape[0]):
            for j in range(transitionMatrix.shape[1]):
                xval = i - k
                yval = j - k
                if i == transitionMatrix.shape[0] // 2:
                    transitionMatrix[i, j] = np.trace(secondMoments, offset= yval)
                else:
                    # This might need to change if the mean of the distribution isn't symmetric
                    transitionMatrix[i, j] = np.trace(firstMoments, offset= xval - yval) 

        # I think need to take the transpose in order 
        # to get the correct inv measure
        transitionMatrix = transitionMatrix.T
        eigenvalues, eigenvectors = linalg.eigs(transitionMatrix, k=1, which='LM')
        argmax = np.argmax(eigenvalues)
        mu = eigenvectors[:,argmax]
        mu = mu / mu[np.argmax(np.abs(mu))]
        
        return mu
    
    alphas = np.array([1, 1, 1, 1, 1])
    size = 501
    mu = getInvMeasure(alphas, size)
    fig, ax = plt.subplots()
    ax.semilogy(np.arange(size) - size // 2, mu)
    ax.set_xlim([-10, 10])
    fig.savefig("InvMeasure.png")


        