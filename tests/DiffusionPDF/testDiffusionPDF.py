import numpy as np
import npquad
import pytest
import sys
import os

sys.path.append("../../src")
from pydiffusionPDF import DiffusionPDF


def test_equals():
    """
    Test the Diffusion equals works correctly.
    """

    d = DiffusionPDF(np.quad("1e4500"), beta=1, occupancySize=100)
    d.id = 0
    d.evolveToTime(100)
    assert d == d

def test_iteratePastOccupancySize():
    """
    Make sure that we can't iterate past the size of the edges (or max time).
    """

    d = DiffusionPDF(np.quad("1e4500"), beta=1, occupancySize=100)
    d.id = 0
    with pytest.raises(RuntimeError) as exinfo:
        d.evolveToTime(101)
    assert "Cannot iterate past the size of the edges" in str(exinfo.value)

def test_pyDiffusion_findQuantiles():
    """
    Check that findQuantile and findQuantiles work the same.
    For beta=inf at t=5, the occupancy will always be:

    [0.03125 0.15625 0.3125 0.3125 0.15625 0.03125]
    """

    diff = DiffusionPDF(1, beta=np.inf, occupancySize=5)
    diff.id = 0
    diff.evolveToTime(5)
    assert diff.findQuantile(10) == 1.5
    assert diff.findQuantile(100) == 2.5
    qs = diff.findQuantiles([100, 10])
    assert all(qs == [2.5, 1.5])


def test_pyDiffusion_findQuantiles_multiplePartilces():
    """
    Check that findQuantile and findQuantiles work the same for nParticles > 1.
    For beta=inf at t=5 the occupancy will always be:

    [0.3125 1.5625 3.125 3.125 1.5625 0.3125]
    """

    diff = DiffusionPDF(10, beta=np.inf, occupancySize=5)
    diff.id = 0
    diff.evolveToTime(5)

    assert diff.findQuantile(10) == 1.5
    assert diff.findQuantile(100) == 2.5
    qs = diff.findQuantiles([100, 10])
    assert all(qs == [2.5, 1.5])


def test_pyDiffusion_findQuantiles_ascendingOrder():
    """
    Check that findQuantiles works if the quantiles are not in ascending order.
    We may want to actually throw an error if they aren't in ascending order or
    check if it's in ascending order and then return the correct quantiles.
    """

    diff = DiffusionPDF(10, beta=np.inf, occupancySize=5)
    diff.id = 0
    diff.evolveToTime(5)

    assert all(diff.findQuantiles([10, 100]) == [2.5, 1.5])
    assert all(diff.findQuantiles([100, 10]) == [2.5, 1.5])


def test_pyDiffusion_evolveAndSaveQuantiles():
    """
    Check that the evolveAndSaveQuantiles function works the same as finding
    quantiles.
    """

    diff = DiffusionPDF(10, beta=np.inf, occupancySize=5)
    diff.id = 0
    diff.evolveAndSaveQuantiles(
        time=[1, 2, 3, 4, 5], quantiles=[100, 10], file="Data.txt"
    )
    data = np.loadtxt("Data.txt", delimiter=",", skiprows=1)
    evolved_quantiles = data[:, [2, 3]]  # Quartiles are the 2nd and 3rd columns

    iterated_quantiles = []
    diff = DiffusionPDF(10, beta=np.inf, occupancySize=5)
    diff.id = 0
    for _ in range(5):
        diff.iterateTimestep()
        iterated_quantiles.append(diff.findQuantiles([100, 10]))

    iterated_quantiles = np.array(iterated_quantiles)

    assert np.all(evolved_quantiles == iterated_quantiles)
    assert np.all([1, 2, 3, 4, 5] == data[:, 0])  # times should be the same

def test_pyDiffusion_ProbDistFlagFalse():
    """
    Check that the ProbDistFlag keeps the same number of particles. Note that
    for larger particles some particles will be lost due to rounding. It's usually
    fairly small compared to the total number of particles though.
    """

    nParticles = np.quad("10")
    diff = DiffusionPDF(nParticles, beta=1, occupancySize=10, ProbDistFlag=False)
    diff.id = 0
    for _ in range(10):
        diff.iterateTimestep()
        assert np.sum(diff.occupancy) == nParticles


def test_pyDiffusion_ProbDistFlagFalse_LargeParticles():
    """
    Check that ProbDistFlag keeps the same number of particles for a large number
    of particles within a certain percent different tolerance.
    """
    tMax = 100
    percent_tolerance = 1e-30
    nParticles = np.quad("1e4500")
    diff = DiffusionPDF(nParticles, beta=1, occupancySize=tMax, ProbDistFlag=False)
    diff.id = 0
    for _ in range(tMax):
        diff.iterateTimestep()
        percent_difference = (np.sum(diff.occupancy) - nParticles) / nParticles
        assert percent_difference < percent_tolerance

def test_pyDiffusion_savedState():
    """
    Check that the variables and occupancy are being properly saved. And that we
    can resize the occupancy and get the same thing back.
    """

    d = DiffusionPDF(100, np.inf, int(1e6), ProbDistFlag=False)
    d.id = 1
    d.evolveToTime(100)
    d.saveState()

    d2 = DiffusionPDF.fromFiles("Scalars1.json", "Occupancy1.txt")

    assert d == d2

def test_pyDiffusion_savedStateIterate():
    """
    Check that the variables save and that we can iterate after loading the
    occupancy.
    """

    nParticles = 1
    tMax = 1000
    diff = DiffusionPDF(nParticles, beta=np.inf, occupancySize=tMax, ProbDistFlag=True)
    diff.id = 1
    diff.evolveToTime(tMax)
    diff.saveState()

    # Load occupancy from the saved variables
    diff2 = DiffusionPDF.fromFiles("Scalars1.json", "Occupancy1.txt")
    diff2.resizeOccupancyAndEdges(5)
    diff2.evolveToTime(diff2.currentTime + 5)

    # Check if it's the same as if we just evolved regularly
    diff3 = DiffusionPDF(nParticles, beta=np.inf, occupancySize=tMax+5, ProbDistFlag=True)
    diff3.id = 1
    diff3.evolveToTime(1000+5)
    assert diff2 == diff3

def remove(file):
    if os.path.exists(file):
        os.remove(file)

def test_cleanup():
    """
    Really just want to delete any files that are still remaining once all the
    tests have been run.
    """
    remove("Data.txt")
    remove("ScalarsNone.json")
    remove("Edges1.json")
    remove("Occupancy1.txt")
    remove("Scalars1.json")
    remove("Scalars0.json")
