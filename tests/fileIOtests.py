import pytest
import os
import numpy as np
import npquad
import sys

sys.path.append("../src/")
sys.path.append("../cDiffusion/")
import fileIO
from pydiffusion import Diffusion


def test_saveArrayQuad_oneDim():
    """
    Test that the array saves properly with a 1d array.
    """

    arr = np.array([np.quad("1e4500"), np.quad("1e-4500"), np.quad("5")], dtype=np.quad)
    fileIO.saveArrayQuad("OneDimData.txt", arr)


def test_saveArrayQuad_multiDim():
    """
    Test that the array saves properly with a multi-Dimensional array.
    """

    arr = np.array([[5, 6], [7, 8]], dtype=np.quad)
    fileIO.saveArrayQuad("MultiDimData.txt", arr)


def test_saveOccupancy():
    """
    Test that we can save the occupancy of our Diffusion object.
    """
    d = Diffusion(1e300, np.inf, 1000)
    d.evolveToTime(1000)
    fileIO.saveArrayQuad("Occ.txt", d.occupancy)
    loaded_array = fileIO.loadArrayQuad("Occ.txt", shape=d.occupancy.shape)
    assert np.all(d.occupancy == loaded_array)


def test_loadArrayQuad_oneDim():
    """
    Test that we can properly load a 1d array. Currently can only do multi-dimensional
    arrays so need to build in 1D.
    """

    arr = np.array([np.quad("1e4500"), np.quad("1e-4500"), np.quad("5")], dtype=np.quad)
    loaded_array = fileIO.loadArrayQuad("OneDimData.txt", shape=3)

    assert np.all(arr == loaded_array)


def test_loadArrayQuad_multiDim():
    """
    Test that we can properly load a multi-dimensional array.
    """

    arr = np.array([[5, 6], [7, 8]], dtype=np.quad)
    loaded_array = fileIO.loadArrayQuad("MultiDimData.txt", shape=(2, 2))
    assert np.all(arr == loaded_array)


def test_teardown():
    """
    Just delete all the files that were created. It's a test b/c I'm lazy.
    """

    files = ["OneDimData.txt", "Occ.txt", "MultiDimData.txt"]
    for f in files:
        os.remove(f)
