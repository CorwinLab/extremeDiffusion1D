import numpy as np
# import npquad
from matplotlib import pyplot as plt

def evolve2DLattice(Length, NParticles, MaxT=None):
    if not MaxT:
        MaxT = Length+1 #automatically tells it the time you can evolve to
    # occupancy = np.zeros((2*Length+1, 2*Length+1)) #create array
    # origin = (Length, Length) #find origin
    # occupancy[origin] = NParticles #place all particles @ origin
    #evolve time
        MaxT = Length+1
    occupancy = np.zeros((2*Length+1, 2*Length+1))
    origin = (Length, Length)
    occupancy[origin] = NParticles
    i,j = np.indices(occupancy.shape)
    checkerboard = (i+j+1) % 2

    for t in range(1,MaxT):
        #generate biases as a txtx4 matrix; (txt) specifies lattice point, the 4 represents probability
        #of going in each direction
        #[[[left,down,right,up]]]
        # Compute biases for every cell within area we're evolving to
        biases = np.random.dirichlet([1]*4, (2*t-1, 2*t-1))

        startPoint = Length-t+1 #define start and end points of the lattice in each dir. for each timestep
        endPoint = Length+t
        oldOccupancy = occupancy[startPoint:endPoint, startPoint:endPoint].copy()
        occupancy[startPoint:endPoint, startPoint-1:endPoint-1] += oldOccupancy * biases[:,:,0]
        occupancy[startPoint+1:endPoint+1, startPoint:endPoint] += oldOccupancy * biases[:,:,1]
        occupancy[startPoint:endPoint, startPoint+1:endPoint+1] += oldOccupancy * biases[:,:,2]
        occupancy[startPoint-1:endPoint-1, startPoint:endPoint] += oldOccupancy * biases[:,:,3]
        occupancy[checkerboard== (t % 2)] = 0
        # # I'm leaving this code here because it does a better job of explaining what our goal is
        # for i in range(startPoint, endPoint):
        #     #across
        #     for j in range(startPoint, endPoint):
        #         # Do the calculation if the site and the time have opposite parity
        #         if (i + j + t) % 2 == 1:
        #             localBiases = biases[i-startPoint, j-endPoint, :]
        #             # left
        #             occupancy[i, j - 1] += occupancy[i, j] * localBiases[0]
        #             # down
        #             occupancy[i + 1, j] += occupancy[i, j] * localBiases[1]
        #             # right
        #             occupancy[i, j + 1] += occupancy[i, j] * localBiases[2]
        #             # up
        #             occupancy[i - 1, j] += occupancy[i, j] * localBiases[3]
        #             # zero the old one
        #             occupancy[i, j] = 0
    return occupancy


