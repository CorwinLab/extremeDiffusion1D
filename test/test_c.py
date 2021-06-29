import pytest
import numpy as np
import sys
sys.path.append('../cDiffusion/')

import cDiffusion as cdiff

def test_neg_occupation():
    '''
    Test handling of negative occupation.
    '''

    occupation = np.array([-10, 0, 5])
    with pytest.raises(RuntimeError) as excinfo:
        edges, occ = cdiff.floatEvolveTimeStep(occupation, beta=1, prevMinIndex=0,
                                                prevMaxIndex=2, N=5)
    assert "One or more variables out of bounds" in str(excinfo.value)

def test_min_greater_max():
    '''
    Test handling of min edge greater than max edge.
    '''
    occupation = np.array([10, 0, 10])
    with pytest.raises(RuntimeError) as excinfo:
        edges, occ  = cdiff.floatEvolveTimeStep(occupation, beta=1, smallCutoff=0,
                                                prevMinIndex=2, prevMaxIndex=0, N=20)
    assert "Minimum edge must be greater than maximum edge" in str(excinfo.value)

def test_max_greater_array_length():
    '''
    Test handling of maxEdgeBound greater than array length.

    Need to check when this actually breaks the for loop!
    '''
    occupation = np.array([10, 10, 10])
    with pytest.raises(IndexError) as excinfo:
        edges, occ = cdiff.floatEvolveTimeStep(occupation, beta=1, prevMinIndex=0,
                                                prevMaxIndex=3, smallCutoff=0, N=30)
    assert "vector::_M_range_check" in str(excinfo.value)

def test_neg_min_edge():
    '''
    Ensure that it throws an error if the minimum edge bound is < 0. Should throw
    an error on the pybind side of things since minEdgeBound is unsigned int.
    '''
    occupation = np.array([10, 10, 10])
    with pytest.raises(TypeError) as excinfo:
        edges, occ = cdiff.floatEvolveTimeStep(occupation, beta=1, prevMinIndex=-1,
                                                prevMaxIndex=2, smallCutoff=0, N=30)
    assert "incompatible function arguments" in str(excinfo.value)

def test_all_zeros():
    '''
    Ensure that it returns all zeros if only zeros input.
    '''
    occupation = np.array([0, 0, 0, 0])
    edges, occupation = cdiff.floatEvolveTimeStep(occupation, 1, 0, 3, 5, 0)
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
    edges, occupied = cdiff.floatEvolveTimeStep(occupation, beta=1, prevMinIndex=0,
                                                prevMaxIndex=1, smallCutoff=5, N=10)
    assert edges[1] == 1, f"Farthest edge is not 1: {occupied}"

def test_smallCutoff_optional():
    '''
    For whatever reason it doesn't seem like Pybind11 likes default arguments
    so going to test why here.
    '''
    occupation = np.array([10, 0, 0])
    edges, occ = cdiff.floatEvolveTimeStep(occupation, beta=1, prevMinIndex=0,
                                            prevMaxIndex=1, N=10)

def test_diffusion_constructor():
    '''
    Make sure the Diffusion object is being initialized correctly. Looks a little
    messy because we want to check all the variables are initialized correctly.
    '''
    d = cdiff.Diffusion(1, 1, 1, 1)

    # Create a list of all the errors that occur
    errors = []

    if not d.getN() == 1:
        errors.append(f"N should be initialized to 1 but is {d.getN()}")
    if not d.getEdges() == ([0], [1]):
        errors.append(f"Edges should be initialzed to (0, 1) but is {d.getEdges()}")
    if not d.getBeta() == 1:
        errors.append(f"Beta should be initialzed to 1.0 but is {d.getBeta()}")
    if not d.getsmallCutoff() == 1:
        errors.append(f"Small cutoff should be initialized to 1 but is {d.getsmallCutoff()}")

    assert not errors, "Errors occured:\n{}".format("\n".join(errors))

def test_diffusion_initializeOccupancyAndEdges():
    '''
    Make sure that the initialization makes everything to length log(N)**5/2
    '''
    N = 1e20
    num_of_elements = round(np.log(N) ** (5/2))
    d = cdiff.Diffusion(N, 1.0)
    d.initializeOccupationAndEdges()
    edges = d.getEdges()
    occ = d.getOccupancy()

    assert (num_of_elements == len(edges[0])
            & num_of_elements == len(edges[1])
            & num_of_elements == len(occ))

def test_diffusion_iterateTimestepInplaceFalse():
    '''
    Make sure iterateTimestep with inplace=False appends edges correctly.
    '''
    N = 1e20
    occ = [N]
    d = cdiff.Diffusion(N, 1.0)
    d.setOccupancy(occ)
    d.iterateTimestep(inplace=False)
    edges = d.getEdges()
    occ = d.getOccupancy()
    num_of_elements = 2
    assert (num_of_elements == len(edges[0])
            and num_of_elements == len(edges[1])
            and num_of_elements+1 == len(occ)) # I think occupancy gets pushback once in iterateTimestep and another time in floatEvolveTimeStep

def test_diffusion_iterateTimestepInplaceTrue():
    '''
    Make sure iterateTimestep with inplace=True works correctly. Does not
    change the size of the occupancy or the edges.
    '''
    N = 1e20
    d = cdiff.Diffusion(N, 1.0)
    d.initializeOccupationAndEdges()
    num_of_elements = len(d.getOccupancy())
    d.iterateTimestep(inplace=True)
    occ = d.getOccupancy()
    minEdge, maxEdge = d.getEdges()
    assert (num_of_elements == len(occ) and num_of_elements == len(minEdge) and num_of_elements == len(maxEdge))

def test_diffusion_iterateTimestepInplaceTrueEdges():
    '''
    Make sure iterateTimestep with inplace=True makes the second edge index nonzero.
    '''
    N = 1e20
    d = cdiff.Diffusion(N, 1.0)
    d.initializeOccupationAndEdges()
    d.iterateTimestep(inplace=True)
    minEdges, maxEdges = d.getEdges()
    nonzeros = np.nonzero(maxEdges)[0]
    assert np.all(nonzeros == [0, 1])

def test_diffusion_evolveTimestepsInplaceTrue():
    '''
    Make sure that evolveTimesteps w/ inplace = True doesn't change the length of the
    edges or occupaction.
    '''
    N = 1e10
    d = cdiff.Diffusion(N, 1.0)
    num_of_steps = round(np.log(N) ** (5/2))
    d.initializeOccupationAndEdges()
    edges_length = len(d.getEdges()[0])
    d.evolveTimesteps(num_of_steps, inplace=True)
    assert (edges_length == len(d.getEdges()[0]))

def test_diffusion_evolveTimestepsInplaceFalse():
    '''
    Make sure that evolveTimesteps w/ inplace = False changes the length of the
    edges and occupation appropriately.
    '''
    N = 1e50
    d = cdiff.Diffusion(N, 1.0)
    num_of_steps = 1000
    d.evolveTimesteps(num_of_steps, inplace=False)

if __name__ == '__main__':
    pytest.main(['./test_c.py'])
