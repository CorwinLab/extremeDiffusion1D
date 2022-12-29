import numpy as np
from matplotlib import pyplot as plt
from numba import njit

def solver(dx=0.1, minmax=(-10, 10), D=1):
    grid = np.arange(minmax[0], minmax[1] + dx, dx)
    f0 = np.zeros(grid.shape)
    f0[len(f0) // 2] = 1
    
    

if __name__ == '__main__':
    solver(stepsize=2)