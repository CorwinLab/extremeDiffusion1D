import sys

sys.path.append("../../src")

from pydiffusionCDF import DiffusionPositionCDF, DiffusionTimeCDF
from nativePyDiffusionCDF import makeRec, findQuintile
import numpy as np
import npquad
import pytest

def test_einsteinbias_CDF():
    """
    Test if the einstein bias case works correctly.
    """

    diff = DiffusionPositionCDF(np.inf, 5)
    totalCDF = [diff.CDF]
    for _ in range(5):
        diff.stepPosition()
        totalCDF.append(diff.CDF)
    totalCDF = np.array(totalCDF).T

    diff = DiffusionTimeCDF(np.inf, 5)
    timeCDF = [diff.CDF]
    for _ in range(5):
        diff.iterateTimeStep()
        timeCDF.append(diff.CDF)
    timeCDF = np.array(timeCDF)
    assert np.all(totalCDF == timeCDF)
