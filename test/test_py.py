import sys

sys.path.append("../src")

import cdiffusion as cdiff
import diffusion as diff
import numpy as np
import pytest

np.random.seed(0)


def test_check_sum_large():
    """
    Make sure that the number of walkers stays the same after one iteration
    for the giant particle cutoff.
    """
    biases = np.array([0.2, 0.7, 1])
    occ = np.array([10, 10, 10])
    new_occ = cdiff.iterate_timestep(occ, biases, smallcutoff=0)
    assert np.sum(occ) == np.sum(new_occ)


def test_all_ones_large():
    """
    Check that with a bias of 1, all the particles move to the right for
    the giant particle cutoff.
    """
    biases = np.array([1, 1, 1])
    occ = np.array([10, 10, 10])
    new_occ = cdiff.iterate_timestep(occ, biases, smallcutoff=0)
    result = np.insert(occ, 0, 0)
    assert np.all(new_occ == result)


def test_all_zeros_large():
    """
    Check that with a bias of 0, all the particles move to the left for
    the giant particle cutoff.
    """
    biases = np.array([0, 0, 0])
    occ = np.array([10, 10, 10])
    new_occ = cdiff.iterate_timestep(occ, biases, smallcutoff=0)
    result = np.append(occ, 0)
    assert np.all(new_occ == result)


def test_einstein_bias_large():
    """
    With a bias of 0.5, the particles should be split 50/50 for the giant cutoff.
    """
    biases = np.array([0.5, 0.5, 0.5])
    occ = np.array([10, 10, 10])
    new_occ = cdiff.iterate_timestep(occ, biases, smallcutoff=0)
    result = np.array([5, 10, 10, 5])
    assert np.all(new_occ == result)


def test_compare_einstein_large():
    """
    Compare einstein bias for the cdiffusion and diffusion algorithms in the giant
    cutoff.

    To compare the two algorithms need to pad the occupancy and biases
    because the original np algorithm doesn't append a value to the end.
    """
    biases = np.array([0.5, 0.5, 0.5, 0.5])
    occ = np.array([10, 10, 10, 0])
    c_occ = cdiff.iterate_timestep(occ, biases, smallcutoff=0)
    c_occ = c_occ[:-1]
    occ = diff.floatEvolveTimeStep(occ, biases, smallCutoff=0)
    assert np.all(c_occ == occ)


def test_compare_random_large():
    """
    Compare the cdiffusion and diffusion algorithms in the giant cutoff for
    random biases and a random occupancy.
    """
    biases = np.random.random(500)
    occ = np.random.random(499) * 100
    occ = np.append(occ, 0)
    occ = np.round(occ)
    c_occ = cdiff.iterate_timestep(occ, biases, smallcutoff=0)
    c_occ = c_occ[:-1]
    reg_occ = diff.floatEvolveTimeStep(occ, biases, smallCutoff=0)
    assert np.all(c_occ == reg_occ)


def test_compare_random_small():
    """
    Compare the small cutoff for random biases and occupancy. Need to figure
    out a way to quantize the differences in randomness between the two tests.
    It looks like they perform the same but just send a different number right/left.
    """
    biases = np.random.random(500)
    occ = np.random.random(499) * 100
    occ = np.append(occ, 0)
    occ = np.round(occ)
    c_occ = cdiff.iterate_timestep(occ, biases, smallcutoff=1000)
    c_occ = c_occ[:-1]
    reg_occ = diff.floatEvolveTimeStep(occ, biases, smallCutoff=1000)
    print("Mean", np.mean(abs(c_occ - reg_occ)))
    assert np.all(c_occ == reg_occ)


if __name__ == "__main__":
    pytest.main(["./test.py", "--disable-warnings"])
