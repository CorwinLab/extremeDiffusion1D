import sys

sys.path.append('../src')

import cdiffusion as bq
import numpy as np
import pytest

def test_check_sum():
    biases = np.array([0.2, 0.7, 1])
    occ = np.array([10,10,10])
    new_occ = bq.iterate_timestep(occ, biases)
    assert np.sum(occ) == np.sum(new_occ)

def test_all_ones():
    biases = np.array([1, 1, 1])
    occ = np.array([10, 10, 10])
    new_occ = bq.iterate_timestep(occ, biases)
    result = np.insert(occ, 0, 0)
    assert np.all(new_occ == result)

def test_all_zeros():
    biases = np.array([0, 0, 0])
    occ = np.array([10, 10, 10])
    new_occ = bq.iterate_timestep(occ, biases)
    result = np.append(occ, 0)
    assert np.all(new_occ == result)

def test_einstein_bias():
    biases = np.array([0.5, 0.5, 0.5])
    occ = np.array([10,10,10])
    new_occ = bq.iterate_timestep(occ, biases)
    result = np.array([5,10,10,5])
    assert np.all(new_occ == result)

pytest.main(["./test.py"])