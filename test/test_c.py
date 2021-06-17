import pytest
import numpy as np
import sys
sys.path.append('../cDiffusion/')

import cDiffusion as cdiff

def test_neg_occupation():
    '''
    Test handling of negative occupation.
    '''

    occupation = np.array([5, 0, -10])
    with pytest.raises(RuntimeError) as excinfo:
        edges, occ = cdiff.floatEvolveTimeStep(occupation, beta=1, smallCutoff=0,
                                                minEdgeIndex=0, maxEdgeIndex=2)
    assert "Occupancy must be > 0" in str(excinfo.value)

def test_min_greater_max():
    '''
    Test handling of min edge greater than max edge.
    '''
    occupation = np.array([10, 0, 10])
    with pytest.raises(RuntimeError) as excinfo:
        edges, occ  = cdiff.floatEvolveTimeStep(occupation, beta=1, smallCutoff=0,
                                                minEdgeIndex=2, maxEdgeIndex=0)
    assert "Minimum edge must be greater than maximum edge" in str(excinfo.value)

def test_max_greater_array_length():
    '''
    Test handling of maxEdgeBound greater than array length.

    Need to check when this actually breaks the for loop!
    '''
    occupation = np.array([10, 10, 10])
    with pytest.raises(RuntimeError) as excinfo:
        edges, occ = cdiff.floatEvolveTimeStep(occupation, beta=1, minEdgeIndex=0,
                                                maxEdgeIndex=3, smallCutoff=0)
    assert "Maximum edge exceeds size of vector" in str(excinfo.value)

def test_neg_min_edge():
    '''
    Ensure that it throws an error if the minimum edge bound is < 0. Should throw
    an error on the pybind side of things since minEdgeBound is unsigned int.
    '''
    occupation = np.array([10, 10, 10])
    with pytest.raises(TypeError) as excinfo:
        edges, occ = cdiff.floatEvolveTimeStep(occupation, beta=1, minEdgeBound=-1,
                                                maxEdgeBound=2, smallCutoff=0)
    assert "incompatible function arguments" in str(excinfo.value)

def test_all_zeros():
    '''
    Ensure that it returns all zeros if only zeros input.
    '''
    occupation = np.array([0, 0, 0, 0])
    edges, occupation = cdiff.floatEvolveTimeStep(occupation, 1, 0, 3, 5)
    zeros = np.array([0, 0, 0, 0, 0])
    assert np.all(occupation == zeros)

def test_single_occupation_filled():
    '''
    Make sure the algorith can accept just a single position filled and distributes
    this to two positions.

    Note:
    -----
    This is somewhat based on chance due to the bias being drawn from a random
    beta distribution. However, it should pass most of the time.
    '''
    occupation = np.array([10, 0, 0])
    edges, occupied = cdiff.floatEvolveTimeStep(occupation, beta=1, minEdgeIndex=0,
                                                maxEdgeIndex=1, smallCutoff=5)
    assert edges[1] == 1

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    pytest.main(['./test_c.py'])
