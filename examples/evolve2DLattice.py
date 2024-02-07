import numpy as np
# import npquad
from matplotlib import pyplot as plt
# from numba import jit, njit

def numpyEvolve2DLattice(length, NParticles, maxT = None, rng = np.random.default_rng()):
    if not maxT:
        maxT = length+1
    occupancy = np.zeros((2*length+1, 2*length+1), dtype=int)
    origin = (length, length)
    occupancy[origin] = int(NParticles)
    i,j = np.indices(occupancy.shape)
    checkerboard = (i+j+1) % 2
    for t in range(1, maxT):
        # Find the occupied sites
        sites = (occupancy != 0)
        i,j = np.where(sites)
        # Generate biases for each site
        biases = rng.dirichlet([1]*4, np.sum(sites))
        # On newer numpy we can vectorize to compute the moves
        moves = rng.multinomial(occupancy[i,j], biases)
        occupancy[i,j-1] += moves[:,0]
        occupancy[i+1,j] += moves[:,1]
        occupancy[i,j+1] += moves[:,2]
        occupancy[i-1,j] += moves[:,3]
        occupancy[i,j] = 0 
        yield t, occupancy
#    return occupancy
    

# # @jit(nopython=True)
# def numbaEvolve2DLattice(length, NParticles, maxT=None):
#     if not maxT:
#         maxT = length+1
#     occupancy = np.zeros((2*length+1, 2*length+1), dtype=np.uint)
#     origin = (length, length)
#     occupancy[origin] = NParticles
#     # i,j = np.indices(occupancy.shape)
#     # checkerboard = (i+j+1) % 2

#     for t in range(1,maxT):
#         for i in range(2*length + 1):
#             for j in range(2*length + 1):
#                 if ((i + j + t) % 2 == 1) and (occupancy[i,j] != 0):
#                     print(i,j,t)
#                     localBias = np.random.exponential(1, size=4)
#                     localBias /= np.sum(localBias)
#                     # moves = localBias * occupancy[i,j]
#                     moves = np.random.multinomial(occupancy[i,j], [.25, .25, .25, .25])
#                     occupancy[i,j-1] += moves[0]
#                     occupancy[i+1,j] += moves[1]
#                     occupancy[i,j+1] += moves[2]
#                     occupancy[i-1,j] += moves[3]
#                     occupancy[i,j] = 0
#     return occupancy


def evolve2DLatticeAgent(Length, NParticles, MaxT=None):
    """
     Create a (2Length+1) square lattice with N particles at the center, and let particles diffuse according to
     dirichlet biases in cardinal directions, with particles/agents according to multinomial distribution (agentbased).
     Parameters:
         Length: distance from origin to side of lattice
         NParticles: number of total particles in system
         MaxT: automatically set to be the maximum time particles can evolve to, but can set a specific time
     """
    # automatically tells it the time you can evolve to
    if not MaxT:
        MaxT = Length + 1
    # initialize the array, particles @ origin, and the checkerboard pattern
    occupancy = np.zeros((2 * Length + 1, 2 * Length + 1))
    origin = (Length, Length)
    occupancy[origin] = NParticles
    i, j = np.indices(occupancy.shape)
    checkerboard = (i + j + 1) % 2
    # evolve in time
    for t in range(1, MaxT):
        # Compute biases for every cell within area we're evolving to
        # [[[left,down,right,up]]]
        biases = np.random.dirichlet([1] * 4, (2 * t - 1, 2 * t - 1))
        # Define interior lattice size to evolve based on timestep
        startPoint = Length - t + 1
        endPoint = Length + t
        # Check each site in lattice and manually move agents to new sites
        for i in range(startPoint, endPoint):
            #across
            for j in range(startPoint, endPoint):
                # Do the calculation if the site and the time have opposite parity
                if (i + j + t) % 2 == 1:
                    # map our set of biases to the sites we need
                    localBiases = biases[i-startPoint, j-endPoint, :]
                    # Use the biases to distribute the agents to the neighboring cells
                    agents = np.random.multinomial(occupancy[i,j], localBiases)
                    # left
                    occupancy[i, j - 1] += agents[0]
                    # down
                    occupancy[i + 1, j] += agents[1]
                    # right
                    occupancy[i, j + 1] += agents[2]
                    # up
                    occupancy[i - 1, j] += agents[3]
                    # zero the old one
                    occupancy[i, j] = 0
        yield t, occupancy

def evolve2DLatticePDF(Length, NParticles, MaxT=None):
    """
    Create a (2Length+1) square lattice with N particles at the center, and let particles diffuse according to
    dirichlet biases in cardinal directions. This evolves the PDF.
    Parameters:
        Length: distance from origin to side of lattice
        NParticles: number of total particles in system
        MaxT: automatically set to be the maximum time particles can evolve to, but can set a specific time
    """
    # automatically tells it the time you can evolve to
    if not MaxT:
        MaxT = Length+1
    # initialize the array, particles @ origin, and the checkerboard pattern
    occupancy = np.zeros((2*Length+1, 2*Length+1))
    origin = (Length, Length)
    occupancy[origin] = NParticles
    i,j = np.indices(occupancy.shape)
    checkerboard = (i+j+1) % 2
    # evolve in time
    for t in range(1,MaxT):
        # Compute biases for every cell within area we're evolving to all at once
        #[[[left,down,right,up]]]
        biases = np.random.dirichlet([1]*4, (2*t-1, 2*t-1))
        # Define interior lattice size to evolve based on timestep
        startPoint = Length-t+1
        endPoint = Length+t
        # save old occupancy, for calculation reasons
        oldOccupancy = occupancy[startPoint:endPoint, startPoint:endPoint].copy()
        # fill occupancy + zero out the old ones
        occupancy[startPoint:endPoint, startPoint-1:endPoint-1] += oldOccupancy * biases[:,:,0]
        occupancy[startPoint+1:endPoint+1, startPoint:endPoint] += oldOccupancy * biases[:,:,1]
        occupancy[startPoint:endPoint, startPoint+1:endPoint+1] += oldOccupancy * biases[:,:,2]
        occupancy[startPoint-1:endPoint-1, startPoint:endPoint] += oldOccupancy * biases[:,:,3]
        occupancy[checkerboard== (t % 2)] = 0
        yield t, occupancy
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
    # return occupancy

# data generating functions
def generateFirstArrivalTime(Length,NParticles,MaxT=None):
    """
    evolves agents in 2Dlattice according to dirichlet, w/ agents moving
    according to multinomial distribution
    parameters:
        Length: # sites from origin to side of the array
        NParticles: # particles in system
        MaxT: optional, automatically set to be the Length+1
    """
    # initialize array to fill with 1st arrival time for each site
    tArrival = np.zeros((2*Length+1,2*Length+1))
    tArrival[:] = np.nan
    for t, occ in evolve2DLatticeAgent(Length,NParticles,MaxT):
        tArrival[(occ > 0) & np.isnan(tArrival)] = t
    return occ, tArrival

# helper functions
#do I really need a wholeass function just to subtract L.
def shiftCoords(i,j,L):
    """ # Turn someone's coordinatess with 0,0 at the center into coords for L,L at center?
    What if i just want 1 coord...
    """
    newcoords = (i-L,j-L)
    return newcoords