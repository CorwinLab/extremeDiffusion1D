import numpy as np
from numba import jit

@jit(nopython=True)
def iteratePDF(pdf, model='RWRE'):
    if model=='RWRE':
        biases = np.random.uniform(0.0, 1.0, size=pdf.shape)
    elif model=='SSRW':
        biases = np.zeros(shape=pdf.shape) + 1/2
    new_pdf = np.zeros(shape=pdf.shape)
    # Deal with the boundaries and then with the rest of the system
    #Absorbing boundary (i == 0)
    new_pdf[0] += pdf[0]
    # The bulk
    for i in range(1,len(pdf)-1):
        new_pdf[i+1] += (1 - biases[i-1]) * pdf[i]
        new_pdf[i-1] += biases[i-1] * pdf[i]
    # Absorbing boundaryelif i==(len(pdf)-1):
    new_pdf[-1] += pdf[-1]

    return new_pdf

def initializePDF(maxPosition):
    pdf = np.zeros(2*maxPosition + 1)
    pdf[maxPosition] = 1
    return pdf
