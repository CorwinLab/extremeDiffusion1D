import sys

sys.path.append("../../src")

from pydiffusionCDF import DiffusionTimeCDF
from nativePyDiffusionCDF import makeRec, findQuintile
import numpy as np
import npquad
import pytest


def test_einsteinbias_CDF():
    """
    Need to test if Eric's code matches what we get from the recurrance relation
    that is implemented in recurrenceRelation/recurrance.cpp
    """

    # Need to do some np magic to make sure Eric's code matches what
    # we're generating. His columns are our rows but reversed.
    CDF = makeRec(5)
    CDF = CDF.T
    for row in range(CDF.shape[0]):
        CDF[row, :] = np.flip(CDF[row, :], axis=0)
        CDF[row, :] = np.roll(CDF[row, :], -CDF.shape[0] + row + 1)

    # Need to save each row
    rec = DiffusionTimeCDF(beta=np.inf, tMax=4)
    CDF_c = rec.CDF
    print(rec.CDF)
    for _ in range(4):
        rec.iterateTimeStep()
        print(rec.CDF)
        CDF_c = np.vstack((CDF_c, rec.CDF))
    assert np.all(CDF_c == CDF)


def test_einsteinbias_CDF_large():
    """
    Test if Eric's code matches for larger values of time. Can't use super large
    values of N b/c we run into precision issues of using floats versus quads.
    """

    N = 50
    CDF = makeRec(N)
    CDF = CDF.T
    for row in range(CDF.shape[0]):
        CDF[row, :] = np.flip(CDF[row, :], axis=0)
        CDF[row, :] = np.roll(CDF[row, :], -CDF.shape[0] + row + 1)

    rec = DiffusionTimeCDF(beta=np.inf, tMax=N - 1)
    CDF_c = rec.CDF
    for _ in range(N - 1):
        rec.iterateTimeStep()
        CDF_c = np.vstack((CDF_c, rec.CDF))

    assert np.all(CDF_c.astype(float) == CDF)


def test_einsteinbias_quartile():
    """
    Test if Eric's code matches when calculating quartiles.
    """

    tMax = 5
    quintile = 10
    CDF = makeRec(tMax)
    qs = findQuintile(CDF, quintile).astype(int)
    rec = DiffusionTimeCDF(beta=np.inf, tMax=tMax - 1)
    qs_c = [rec.findQuantile(quintile)]
    for _ in range(tMax - 1):
        rec.iterateTimeStep()
        qs_c.append(rec.findQuantile(quintile))
    assert (qs == qs_c).all()


def test_einsteinbias_quartile_large():
    """
    Test if Eric's code matches when calculating quartiles at large times.
    """

    tMax = 1000
    quintile = 100
    CDF = makeRec(tMax)
    qs = findQuintile(CDF, quintile).astype(int)
    rec = DiffusionTimeCDF(beta=np.inf, tMax=tMax - 1)
    qs_c = [rec.findQuantile(quintile)]
    for _ in range(tMax - 1):
        rec.iterateTimeStep()
        qs_c.append(rec.findQuantile(quintile))
    assert (qs == qs_c).all()
