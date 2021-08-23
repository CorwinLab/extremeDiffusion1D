import sys

sys.path.append("../../src")

from pydiffusionCDF import DiffusionCDF
from nativePyDiffusionCDF import makeRec, findQuintile
import numpy as np
import npquad
import pytest


def test_einsteinbias_zB():
    """
    Need to test if Eric's code matches what we get from the recurrance relation
    that is implemented in recurrenceRelation/recurrance.cpp
    """

    # Need to do some np magic to make sure Eric's code matches what
    # we're generating. His columns are our rows but reversed.
    zB = makeRec(5)
    zB = zB.T
    for row in range(zB.shape[0]):
        zB[row, :] = np.flip(zB[row, :], axis=0)
        zB[row, :] = np.roll(zB[row, :], -zB.shape[0] + row + 1)

    # Need to save each row
    rec = DiffusionCDF(beta=np.inf, tMax=4)
    zB_c = rec.zB
    print(rec.zB)
    for _ in range(4):
        rec.iterateTimeStep()
        print(rec.zB)
        zB_c = np.vstack((zB_c, rec.zB))
    assert np.all(zB_c == zB)


def test_einsteinbias_zB_large():
    """
    Test if Eric's code matches for larger values of time. Can't use super large
    values of N b/c we run into precision issues of using floats versus quads.
    """

    N = 50
    zB = makeRec(N)
    zB = zB.T
    for row in range(zB.shape[0]):
        zB[row, :] = np.flip(zB[row, :], axis=0)
        zB[row, :] = np.roll(zB[row, :], -zB.shape[0] + row + 1)

    rec = DiffusionCDF(beta=np.inf, tMax=N - 1)
    zB_c = rec.zB
    for _ in range(N - 1):
        rec.iterateTimeStep()
        zB_c = np.vstack((zB_c, rec.zB))

    assert np.all(zB_c.astype(float) == zB)


def test_einsteinbias_quartile():
    """
    Test if Eric's code matches when calculating quartiles.
    """

    tMax = 5
    quintile = 10
    zB = makeRec(tMax)
    qs = findQuintile(zB, quintile).astype(int)
    rec = DiffusionCDF(beta=np.inf, tMax=tMax - 1)
    qs_c = [rec.findQuantile(quintile)]
    for _ in range(tMax - 1):
        rec.iterateTimeStep()
        qs_c.append(rec.findQuantile(quintile))
    assert (qs == qs_c).all()


def test_einsteinbias_quartile_large():
    """
    Test if Eric's code matches when calculating quartiles at large times.
    """

    tMax = 50
    quintile = 10
    zB = makeRec(tMax)
    qs = findQuintile(zB, quintile).astype(int)
    rec = DiffusionCDF(beta=np.inf, tMax=tMax - 1)
    qs_c = [rec.findQuantile(quintile)]
    for _ in range(tMax - 1):
        rec.iterateTimeStep()
        qs_c.append(rec.findQuantile(quintile))
    assert (qs == qs_c).all()
