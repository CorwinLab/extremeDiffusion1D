import numpy as np
# import npquad
from matplotlib import pyplot as plt

def evolve2DLattice(Length, NParticles, MaxT=None):
    if not MaxT:
        MaxT = Length+1
    occupancy = np.zeros((2*Length+1, 2*Length+1))
    origin = (Length, Length)
    occupancy[origin] = NParticles
    for t in range(1,MaxT):
        #generate biases as a txtx4 matrix; (txt) specifies lattice point, the 4 represents probability
        #of going in each direction
        #[[[left,down,right,up]]]
        # Compute biases for every cell
        biases = np.random.dirichlet([1]*4, (2*t-1, 2*t-1))

        startPoint = Length-t+1
        endPoint = Length+t
        for i in range(startPoint, endPoint):
            #across
            for j in range(startPoint, endPoint):
                # Do the calculation if the site and the time have opposite parity
                if (i + j + t) % 2 == 1:
                    localBiases = biases[i-startPoint, j-endPoint, :]
                    # left
                    occupancy[i, j - 1] += occupancy[i, j] * localBiases[0]
                    # down
                    occupancy[i + 1, j] += occupancy[i, j] * localBiases[1]
                    # right
                    occupancy[i, j + 1] += occupancy[i, j] * localBiases[2]
                    # up
                    occupancy[i - 1, j] += occupancy[i, j] * localBiases[3]
                    # zero the old one
                    occupancy[i, j] = 0
    return occupancy


