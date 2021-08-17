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
    assert np.all(d2.occupancy[:1001] == d.occupancy), "Occupany was not initialized correctly."
