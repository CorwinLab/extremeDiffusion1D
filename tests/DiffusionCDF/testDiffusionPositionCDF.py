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

    diff = DiffusionPositionCDF(np.inf, 5, [10])
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

def test_einsteinbias_CDF_findquantile():
    """
    Test if the einstein bias finding quantile works correctly.
    """
    tMax = 10
    diff = DiffusionPositionCDF(np.inf, tMax, [10])
    for _ in range(tMax):
        diff.stepPosition()

    diff2 = DiffusionTimeCDF(np.inf, tMax)
    quantile_position = [diff2.findQuantile(10)]
    for _ in range(tMax):
        diff2.iterateTimeStep()
        quantile_position.append(diff2.findQuantile(10))

    assert np.all(diff.getQuantilesMeasurement()[0] == quantile_position)
