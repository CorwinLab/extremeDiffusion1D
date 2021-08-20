import numpy as np
import npquad
import pytest
import sys

sys.path.append("../src")
from pydiffusion import Diffusion


def test_equals():
    """
    Test the Diffusion equals works correctly.
    """

    d = Diffusion(np.quad("1e4500"), beta=1, occupancySize=100)
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
    d = Diffusion(N, beta, time_steps, probDistFlag)
    d.evolveToTime(time_steps)

    d2 = Diffusion.fromOccupancyTime(
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
    d = Diffusion(N, beta, time_steps, probDistFlag)
    d.evolveToTime(time_steps)

    d2 = Diffusion.fromOccupancyTime(
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

    diff = Diffusion(1, beta=np.inf, occupancySize=5)
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

    diff = Diffusion(10, beta=np.inf, occupancySize=5)
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

    diff = Diffusion(10, beta=np.inf, occupancySize=5)
    diff.evolveToTime(5)

    assert diff.findQuantiles([10, 100]) == [2.5, 1.5]
    assert diff.findQuantiles([100, 10]) == [2.5, 1.5]
