from pyDiffusion.pydiffusionPDF import DiffusionPDF
import numpy as np
# import npquad
from matplotlib import pyplot as plt

#testing to build uup 2D ? oh maybe I should do this in a different folder.
# size=5
# NParticles = 5
# occupancy = np.zeros([size,size])
# occupancy[int((size - 1) / 2), int((size - 1) / 2)] = NParticles
# t = 1
#
# # generate biases as a txtx4 matrix; (txt) specifies lattice point, the 4 represents probability
#     # of going in each direction
#     # [[[left,down,right,up]]]
# biases = np.random.dirichlet(4 * [1], [t, t])
#     # fill with particles
#     # left ??? also biases[:,:,0] says take no elements from txt, the 0th element from the last dimension
# occupancy[2, 1] += occupancy[2, 2] * biases[:, :, 0]
#     # down
# occupancy[3, 2] += occupancy[2, 2] * biases[:, :, 1]
#     # right
# occupancy[2, 3] += occupancy[2, 2] * biases[:, :, 2]
#     # up
# occupancy[1, 2] += occupancy[2, 2] * biases[:, :, 3]
#     # zero out the old spot
# occupancy[2, 2] = 0
# print(occupancy)


# # I think this one is not correct
# def Evolve2DLattice(Length,NParticles,MaxT=2):
#     occupancy = np.zeros([2*Length+1,2*Length+1])
#     origin = Length,Length
#     occupancy[origin] = NParticles
#     for t in range(1,MaxT):
#         #generate biases as a txtx4 matrix; (txt) specifies lattice point, the 4 represents probability
#         #of going in each direction
#         #[[[left,down,right,up]]]
#         biases = np.random.dirichlet(4 * [1], [t, t])
#         # t=0 @ origin, t=1 at odd, t=2 at even, etc.
#
#         #fill with particles
#         #down
#         print("t = ",t)
#         for i in range(1,2*Length):
#             #across
#             for j in range(1,2*Length):
#                 print(" i,j ", i,j, " i + j =",i+j)
#                 #even
#                 if (i+j)%2 == 0:
#                     # left
#                     occupancy[i, j - 1] += occupancy[i, j] * biases[:, :, 0]
#                     # down
#                     occupancy[i + 1, j] += occupancy[i, j] * biases[:, :, 1]
#                     # right
#                     occupancy[i, j + 1] += occupancy[i, j] * biases[:, :, 2]
#                     # doown
#                     occupancy[i - 1, j] += occupancy[i, j] * biases[:, :, 3]
#                     # zero the old one
#                     occupancy[i, j] = 0
#                 #odd
#                 else:
#                     # left
#                     occupancy[i, j - 1] += occupancy[i, j] * biases[:, :, 0]
#                     # down
#                     occupancy[i + 1, j] += occupancy[i, j] * biases[:, :, 1]
#                     # right
#                     occupancy[i, j + 1] += occupancy[i, j] * biases[:, :, 2]
#                     # down
#                     occupancy[i - 1, j] += occupancy[i, j] * biases[:, :, 3]
#                     # zero the old one
#                     occupancy[i, j] = 0
#     return occupancy
#
#
# I think this one is not correct
def Evolve2DLattice(Length,NParticles,MaxT=2):
    occupancy = np.zeros([2*Length+1,2*Length+1])
    origin = Length,Length
    occupancy[origin] = NParticles
    for t in range(1,MaxT):
        #generate biases as a txtx4 matrix; (txt) specifies lattice point, the 4 represents probability
        #of going in each direction
        #[[[left,down,right,up]]]
        biases = np.random.dirichlet(4 * [1], [t, t])
        print(biases)
        #fill with particles
        #down
        print("t = ",t)

        for i in range(1,2*Length):
            #across
            for j in range(1,2*Length):
                print(" i,j ", i,j, " i + j =",i+j, "Conditions: ",(i+j)%2 != 0 and (t % 2 !=0))
                #even
                if (i+j)%2 == 0 and (t % 2 !=0):
                    print('occupancy[i,j-1] and shape: ',occupancy[i,j-1],occupancy[i,j-1].shape)
                    print('biases[:,:,0] shape: ',biases[:,:,0],biases[:,:,0].shape)
                    # left
                    occupancy[i, j - 1] += occupancy[i, j] * biases[:, :, 0]
                    # down
                    occupancy[i + 1, j] += occupancy[i, j] * biases[:, :, 1]
                    # right
                    occupancy[i, j + 1] += occupancy[i, j] * biases[:, :, 2]
                    # doown
                    occupancy[i - 1, j] += occupancy[i, j] * biases[:, :, 3]
                    # zero the old one
                    occupancy[i, j] = 0
                #odd
                elif ((i+j)%2 != 0) and (t % 2 == 0):
                    print('occupancy[i,j-1] and shape: ',occupancy[i,j-1],occupancy[i,j-1].shape)
                    print('biases[:,:,0] shape: ',biases[:,:,0],biases[:,:,0].shape)
                    # left
                    occupancy[i, j - 1] += occupancy[i, j] * biases[:, :, 0]
                    # down
                    occupancy[i + 1, j] += occupancy[i, j] * biases[:, :, 1]
                    # right
                    occupancy[i, j + 1] += occupancy[i, j] * biases[:, :, 2]
                    # down
                    occupancy[i - 1, j] += occupancy[i, j] * biases[:, :, 3]
                    # zero the old one
                    occupancy[i, j] = 0
    return occupancy


