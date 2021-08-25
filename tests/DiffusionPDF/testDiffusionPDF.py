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
    d.evolveToTime(100)
    assert d == d


def test_pyDiffusion_fromOccupancyTime():
    """
    Test to see if the fromOccupancyTime method correctly returns the same object
    that it was initialized from.
    """
    N = np.quad("1e4500")
    beta = 1
    time_steps = 1000
    probDistFlag = True
    d = DiffusionPDF(N, beta, time_steps, probDistFlag)
    d.evolveToTime(time_steps)

    d2 = DiffusionPDF.fromOccupancyTime(
        beta,
        N,
        resize=0,
        time=d.currentTime,
        occupancy=d.occupancy,
        probDistFlag=probDistFlag,
    )
    assert d == d2


def test_pyDiffusion_fromOccupancyTime_resize():
    """
    Test to see if hte fromOccupancyTime method correctly resizes the occupancy.
    """

    N = np.quad("1e4500")
    beta = 1
    time_steps = 1000
    probDistFlag = True
    d = DiffusionPDF(N, beta, time_steps, probDistFlag)
    d.evolveToTime(time_steps)

    d2 = DiffusionPDF.fromOccupancyTime(
        beta,
        N,
        resize=1000,
        time=d.currentTime,
        occupancy=d.occupancy,
        probDistFlag=probDistFlag,
    )

    assert len(d2.occupancy) == 2001, "Occupancy was not resized correctly"
    assert np.all(
        d2.occupancy[:1001] == d.occupancy
    ), "Occupany was not initialized correctly."


def test_pyDiffusion_findQuantiles():
    """
    Check that findQuantile and findQuantiles work the same.
    For beta=inf at t=5, the occupancy will always be:

    [0.03125 0.15625 0.3125 0.3125 0.15625 0.03125]
    """

    diff = DiffusionPDF(1, beta=np.inf, occupancySize=5)
    diff.evolveToTime(5)
    assert diff.findQuantile(10) == 1.5
    assert diff.findQuantile(100) == 2.5
    qs = diff.findQuantiles([100, 10])
    qs.reverse()
    assert qs == [1.5, 2.5]


def test_pyDiffusion_findQuantiles_multiplePartilces():
    """
    Check that findQuantile and findQuantiles work the same for nParticles > 1.
    For beta=inf at t=5 the occupancy will always be:

    [0.3125 1.5625 3.125 3.125 1.5625 0.3125]
    """

    diff = DiffusionPDF(10, beta=np.inf, occupancySize=5)
    diff.evolveToTime(5)

    assert diff.findQuantile(10) == 1.5
    assert diff.findQuantile(100) == 2.5
    qs = diff.findQuantiles([100, 10])
    assert qs == [2.5, 1.5]


def test_pyDiffusion_findQuantiles_ascendingOrder():
    """
    Check that findQuantiles works if the quantiles are not in ascending order.
    We may want to actually throw an error if they aren't in ascending order or
    check if it's in ascending order and then return the correct quantiles.
    """

    diff = DiffusionPDF(10, beta=np.inf, occupancySize=5)
    diff.evolveToTime(5)

    assert diff.findQuantiles([10, 100]) == [2.5, 1.5]
    assert diff.findQuantiles([100, 10]) == [2.5, 1.5]


def test_pyDiffusion_evolveAndSaveQuantiles():
    """
    Check that the evolveAndSaveQuantiles function works the same as finding
    quantiles.
    """

    diff = DiffusionPDF(10, beta=np.inf, occupancySize=5)
    diff.evolveAndSaveQuantiles(
        time=[1, 2, 3, 4, 5], quantiles=[100, 10], file="Data.txt"
    )
    data = np.loadtxt("Data.txt", delimiter=",", skiprows=1)
    evolved_quantiles = data[:, [2, 3]]  # Quartiles are the 2nd and 3rd columns

    iterated_quantiles = []
    diff = DiffusionPDF(10, beta=np.inf, occupancySize=5)
    for _ in range(5):
        diff.iterateTimestep()
        iterated_quantiles.append(diff.findQuantiles([100, 10]))

    iterated_quantiles = np.array(iterated_quantiles)

    assert np.all(evolved_quantiles == iterated_quantiles)
    assert np.all([1, 2, 3, 4, 5] == data[:, 0])  # times should be the same


def test_pyDiffusion_probDistFlagFalse():
    """
    Check that the probDistFlag keeps the same number of particles. Note that
    for larger particles some particles will be lost due to rounding. It's usually
    fairly small compared to the total number of particles though.
    """

    nParticles = np.quad("10")
    diff = DiffusionPDF(nParticles, beta=1, occupancySize=10, probDistFlag=False)
    for _ in range(10):
        diff.iterateTimestep()
        assert np.sum(diff.occupancy) == nParticles


def test_cleanup():
    """
    Really just want to delete any files that are still remaining once all the
    tests have been run.
    """
    if os.path.exists("Data.txt"):
        os.remove("Data.txt")
