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
        edges = cdiff.floatEvolveTimeStep(occupation, beta=1, smallCutoff=0,
                                            minEdgeBound=0, maxEdgeBound=2)
    assert "Occupancy must be > 0" in str(excinfo.value)

def test_min_greater_max():
    '''
    Test handling of min edge greater than max edge.
    '''
    occupation = np.array([10, 0, 10])
    with pytest.raises(RuntimeError) as excinfo:
        edges = cdiff.floatEvolveTimeStep(occupation, beta=1, smallCutoff=0,
                                            minEdgeBound=2, maxEdgeBound=0)
    assert "Minimum edge must be greater than maximum edge" in str(excinfo.value)

def test_max_greater_array_length():
    '''
    Test handling of maxEdgeBound greater than array length.

    Need to check when this actually breaks the for loop!
    '''
    occupation = np.array([10, 10, 10])
    with pytest.raises(RuntimeError) as excinfo:
        cdiff.floatEvolveTimeStep(occupation, beta=1, smallCutoff=0,
                                    minEdgeBound=0, maxEdgeBound=3)
    assert "Maximum edge exceeds size of array" in str(excinfo.value)

def test_neg_min_edge():
    '''
    Ensure that it throws an error if hte minimum edge bound is < 0.
    '''
    occupation = np.array([10, 10, 10])
    with pytest.raises(RuntimeError) as excinfo:
        cdiff.floatEvolveTimeStep(occupation, beta=1, smallCutoff=0,
                                    minEdgeBound=-1, maxEdgeBound=2)
    assert "Minimum edge must be >= 0" in str(excinfo.value)

def test_all_zeros():
    '''
    Ensure that it returns all zeros if only zeros input.
    '''
    occupation = np.array([0, 0, 0, 0])
    cdiff.floatEvolveTimeStep(occupation, beta=1, smallCutoff=0,
                                minEdgeBound=0, maxEdgeBound=4)
    zeros = np.array([0, 0, 0, 0])
    assert occupation == zeros
