# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 16:14:55 2021

@author: jacob
"""

import numpy as np

def iterate_timestep(occ, biases):

    new_occ = np.zeros(len(occ) + 1)
    left_shift = 0

    for i in range(len(occ)):
        right_shift = int(occ[i] * biases[i])

        if i==0:
            new_occ[i] = occ[i] - right_shift
            left_shift = right_shift
            continue

        if i == (len(occ) - 1):
            new_occ[i + 1] = right_shift

        new_occ[i] = occ[i] - right_shift + left_shift
        left_shift = right_shift

    return new_occ

if __name__ == '__main__':
    biases = np.array([0.2, 0.7, 1])
    occ = np.array([10,10,10])
    new_occ = iterate_timestep(occ, biases)
    assert np.sum(occ) == np.sum(new_occ)