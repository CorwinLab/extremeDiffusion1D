import numpy as np
# import npquad
from matplotlib import pyplot as plt
import csv
# from numba import jit, njit

def doubleOccupancyArray(occupancy):
    # This assumes that occupancy is square and with odd size
    length = (occupancy.shape[0])//2
    if length == 0:
        newLength = 1
    else:
        newLength = 2 * length
    newOccupancy = np.zeros((2 * newLength + 1, 2 * newLength + 1), dtype=int)
    newOccupancy[newLength-length:newLength+length+1, newLength-length:newLength+length+1] = occupancy
    return newOccupancy

def executeMoves(occupancy, i, j, rng):
    # Generate biases for each site
    biases = rng.dirichlet([1]*4, i.shape[0])
    # On newer numpy we can vectorize to compute the moves
    moves = rng.multinomial(occupancy[i,j], biases)
    # Note that we can use the same array because we're doing checkerboard moves
    # If we want to use a more general jump kernel we need to use a new (empty) copy of the space
    occupancy[i,j-1] += moves[:,0] # left
    occupancy[i+1,j] += moves[:,1] # down
    occupancy[i,j+1] += moves[:,2] # right
    occupancy[i-1,j] += moves[:,3] # up
    occupancy[i,j] = 0 # Remove everything from the original site, as it's moved to new sites
    return occupancy

def numpyEvolve2DLatticeAgent(occupancy, maxT, startT = 1, rng = np.random.default_rng()):
    # Convert a scalar occupancy into a 2d array of size (1,1)
    if np.isscalar(occupancy):
        occupancy = np.array([[occupancy]], dtype=int)
    for t in range(startT, maxT):
        # Find the occupied sites
        i,j = np.where(occupancy != 0)
        # If the occupied sites are at the limits (i.e if min(i,j) = 0 or max(i,j) = size)
        # then we need to enlarge occupancy and create a new array
        # print(f'{occupancy.shape}, {np.min([i,j])}, {np.max([i,j])}')
        if (np.min([i,j]) <= 0) or (np.max([i,j]) >= np.min(occupancy.shape) -1 ):
            occupancy = doubleOccupancyArray(occupancy)
            # These next two lines are a waste and we could just do index translation
            sites = (occupancy != 0)
            i,j = np.where(sites)
        occupancy = executeMoves(occupancy, i, j, rng)
        yield t, occupancy

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

def evolve2DLatticePDF(Length, MaxT=None):
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
    occupancy[origin] = 1
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
    for t, occ in numpyEvolve2DLattice(Length,NParticles,MaxT):
        tArrival[(occ > 0) & np.isnan(tArrival)] = t
    return occ, tArrival

# helper functions
#do I really need a wholeass function just to subtract L.
def cartToPolar(i,j):
    """
    Can take indices (i,j) and turn them into polar coords. r, theta
    Note: indices need to be already shifted so origin is at center appropriately
    """
    r = np.sqrt(i**2+j**2)
    theta = np.arctan2(j,i)
    return r, theta

#should i turn this into a function? the check how circular
# the mean tArrivals are?
def checkIfMeanTCircular(meanTArrival,band):
    """
    Takes an array of meanTArrival, chooses a band of TArrival
    and plots the coordinates of the meanTArrivals in the band as
    polar coords, i.e plots theta, r.
    If radially symmetric (circularly?) then should get a flat line
    Parameters
    meanTArrival: should be like np.mean(tArrival,0) where tArrival
        is like (#runs,2L+1,2L+1) array. shape of meanTArrival
        should be (2L+1,2L+1)
    band: [lower,upper] of meanTArrival
    """
    cond = ((band[0]<meanTArrival) & (meanTArrival<band[1]))
    L = int(((meanTArrival.shape[0])-1)/2)
    i,j = np.where(cond)
    # this is the stupidest way of extracting L
    # anyway it shifts coords so oriign @ center
    i, j = i-L,j-L
    r, theta = cartToPolar(i,j)
    fig, ax = plt.subplots()
    ax.set_xlabel("Theta")
    ax.set_ylabel("Distance to Center")
    ax.plot(theta,r,'.')
    plt.show()

def plotVarTvsDistance(varT,powerlaw=0):
    """
    Plots the variance of tArrival as a function of distance from origin
    on a loglog scale
    Parameters:
        varT: array (2L+1,2L+1) in size. should come from like np.nanvar(tArrival,0)
        powerlaw: automatically set to 0, if not 0 then can also plot a guessed powerlaw
    """
    L = int(((varT.shape[0])-1)/2)
    i, j = np.meshgrid(range(varT.shape[0]),range(varT.shape[0]))
    i, j = i - L, j - L
    r, theta = cartToPolar(i,j)
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlabel("Distance from origin")
    ax.set_ylabel("Var(TArrival)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(r.flatten(),varT.flatten(),'.')
    if powerlaw != 0:
        x=np.logspace(1,2)
        ax.plot(x,1e-3*x**powerlaw)
    plt.show()

def tArrivalPastPlane(tArrival,line,axis):
    """
    Define a plane, and ask for the first tArrival past that plane
    it should be the smallest tArrival *on* the line*
    Parameters:
        tArrival: an individual run of generateFirstArrivalTime, and
            be a (2L+1,2L+1) shape
        line: idk i need to figure out how to define a line
        axis: i or j; if i then the line drawn is i=line, if j then line is j=line
    """
    if axis == 'i':
        # choose all sites with i=line
        sites = tArrival[line,:]
    elif axis == 'j':
        sites = tArrival[:,line]
    #find the minimum tArrival value in that set of sites
    firstCrossing = np.nanmin(sites)
    return firstCrossing

def getPDFAtRadius(occ, r):
    x = np.arange(-(occ.shape[0] // 2), occ.shape[0] // 2 + 1)
    xx, yy = np.meshgrid(x, x)

    dist_from_center = np.sqrt(xx**2 + yy**2)
    all_indeces = np.where(dist_from_center == r)
    return occ[all_indeces]

def measurePDFatRad(tMax, save_file, r):

    f = open(save_file, 'a')
    writer = csv.writer(f)

    for t, occ in evolve2DLatticePDF(tMax, tMax):
        probs = getPDFAtRadius(occ, r)
        if np.sum(probs) == 0:
            continue
        writer.writerow([t, *probs])

    f.close()