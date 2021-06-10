import sys

sys.path.append('../src')

import cdiffusion as cdiff
import diffusion as diff
import numpy as np
import pytest

np.random.seed(0)

def test_check_sum_large():
    biases = np.array([0.2, 0.7, 1])
    occ = np.array([10,10,10])
    new_occ = cdiff.iterate_timestep(occ, biases, smallcutoff=0)
    assert np.sum(occ) == np.sum(new_occ)

def test_all_ones_large():
    biases = np.array([1, 1, 1])
    occ = np.array([10, 10, 10])
    new_occ = cdiff.iterate_timestep(occ, biases, smallcutoff=0)
    result = np.insert(occ, 0, 0)
    assert np.all(new_occ == result)

def test_all_zeros_large():
    biases = np.array([0, 0, 0])
    occ = np.array([10, 10, 10])
    new_occ = cdiff.iterate_timestep(occ, biases, smallcutoff=0)
    result = np.append(occ, 0)
    assert np.all(new_occ == result)

def test_einstein_bias_large():
    biases = np.array([0.5, 0.5, 0.5])
    occ = np.array([10,10,10])
    new_occ = cdiff.iterate_timestep(occ, biases, smallcutoff=0)
    result = np.array([5,10,10,5])
    assert np.all(new_occ == result)

def test_compare_einstein():
    '''
    To compare the two algorithms need to pad the occupancy and biases
    because the original np algorithm doesn't append a value to the end.
    '''

    biases = np.array([0.5, 0.5, 0.5, 0.5])
    occ = np.array([10, 10, 10, 0])
    c_occ = cdiff.iterate_timestep(occ, biases, smallcutoff=3)
    c_occ = c_occ[:-1]
    occ = diff.floatEvolveTimeStep(occ, biases, smallCutoff=3)
    assert np.all(c_occ == occ)

def test_compare_random_large():
    biases = np.random.random(500)
    occ = np.random.random(499) * 100
    occ = np.append(occ, 0)
    occ = np.round(occ)
    c_occ = cdiff.iterate_timestep(occ, biases, smallcutoff=0)
    c_occ = c_occ[:-1]
    reg_occ = diff.floatEvolveTimeStep(occ, biases, smallCutoff=0)
    assert np.all(c_occ == reg_occ)

def test_compare_random_small():
    biases = np.random.random(500)
    occ = np.random.random(499) * 100
    occ = np.append(occ, 0)
    occ = np.round(occ)
    c_occ = cdiff.iterate_timestep(occ, biases, smallcutoff=0)
    c_occ = c_occ[:-1]
    reg_occ = diff.floatEvolveTimeStep(occ, biases, smallCutoff=1000)
    assert np.all(c_occ == reg_occ)

if __name__ == '__main__':
    pytest.main(['./test.py', "--disable-warnings"])
