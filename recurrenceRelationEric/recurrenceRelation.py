from numba import jit
import numpy as np


@jit(nopython=True)
def makeRec(tMax):
    # Place a one on every diagonal entry
    # zB[n,t]
    zB = np.zeros((tMax,tMax))
    for n in range(tMax):
        for t in range(n,tMax):
            bias = 0.5
            if n == t:
                zB[n,t] = 1
            elif n == 0:
                zB[n,t] = zB[n,t-1]*bias
            else:
                zB[n,t] = zB[n,t-1]*bias + zB[n-1,t-1]*(1-bias)
            # print(n,t, zB[n,t])

    return zB

# @jit(nopython=True)
# def findQuintile(tMax, N):
#     # Place a one on every diagonal entry
#     # zB[n,t]
#     zB = np.zeros(tMax)
#     zBOld = np.zeros(tMax)
#     quintile = np.zeros(tMax)
#
#     for n in range(tMax):
#         for t in range(n,tMax):
#             bias = np.random.rand()
#             if n == t:
#                 zB[n] = 1
#             elif n == 0:
#                 zB[n] = zBOld[n]*bias
#             else:
#                 zB[n] = zBOld[n]*bias + zBOld[n-1]*(1-bias)
#             # print(n,t, zB[n,t])
#             zBOld[n] = zB[n]
#     return zB


@jit(nopython=True)
def findQuintile(zB,N):
    tMax = zB.shape[0]
    quintile = np.zeros(tMax)
    for t in range(tMax):
        quintile[t] = t - 2*np.where(zB[:,t] > (1/N))[0][0] + 2
        n = np.where(zB[:, t] > (1/N))[0][0]
        print("n=", n, " t=", t)
    return quintile
