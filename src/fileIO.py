# @Author: Eric Corwin <ecorwin>
# @Date:   2018-12-18T12:14:58-08:00
# @Email:  eric.corwin@gmail.com
# @Filename: fileIO.py
# @Last modified by:   ecorwin
# @Last modified time: 2019-01-22T17:03:17-08:00


# -*- coding: utf-8 -*-
"""
pyCudaPacking
File Input/Output Functions
Created 17 July 2017
"""
import numpy as np
import npquad
from scipy import sparse, io


def saveArray(fileName, array):
    """
    Saves a 1D array of strings of string coercible types as a tab separated
    file

    args:
        fileName (str): name of file t write
        array (1D array of string coercible objects): array to be saved
        sep (character): separator. for maximum compatibility, should be
        recognized by split() as whitespace.
    """
    with open(fileName, "w") as file:
        for entry in array:
            file.write(str(entry) + "\n")


def loadArray(fileName, dtype):
    """
    Loads a whitespace separated file onto a 1D numpy array of specified
    dtype.

    args:
        fileName (str): name of file t write
        dtype (type): type of output, constructor must accept strings
    """
    with open(fileName, "r") as file:
        array = np.empty(sum(1 for line in file), dtype=dtype)
    with open(fileName, "r") as file:
        for i, line in enumerate(file):
            array[i] = dtype(line.strip())
    return array


def save2DArray(fileName, array, sep="\t"):
    """
    Saves a 2D array of strings of string coercible types as a tab separated
    file

    args:
        fileName (str): name of file to write
        array (2D array of string coercible objects): array to be saved
        sep (character): separator. for maximum compatibility, should be
        recignized by split() as whitespace.
    """
    with open(fileName, "w") as file:
        for line in array:
            file.write(sep.join(map(str, line)) + "\n")


def load2DArray(fileName, dtype):
    """
    loads a whitespace separated file onto a 2D numpy array
    with a chosen type

    args:
        fileName (str): name of file to read
        dtype (type): type of output, constructor must accept strings
    """
    with open(fileName, "r") as file:
        line = file.__next__()
        array = np.empty((1 + sum(1 for line in file), len(line.split())), dtype=dtype)
    with open(fileName, "r") as file:
        for i, line in enumerate(file):
            for j, element in enumerate(line.split()):
                array[i, j] = dtype(element)
    return array


def saveNDArray(fileName, array, sep="\t"):
    """
    Saves a array of strings of string coercible types as a tab separated
    file

    Works for arbitray dimension, but previous cases are preferred for 1-2D

    args:
        fileName (str): name of file to write
        array (numpy.ndarray of string coercible objects): array to be saved
        sep (character): separator. for maximum compatibility, should be
        recignized by split() as whitespace.
    """
    with open(fileName, "w") as file:
        file.write(sep.join(map(str, array.shape)) + "\n")
        for entry in array.flat:
            file.write(str(entry) + "\n")


def loadNDArray(fileName, dtype):
    """
    loads a whitespace separated file onto a numpy.ndarray
    with a chosen dtype

    args:
        fileName (str): name of file to read
        dtype (type): type of output, constructor must accept strings
    """
    with open(fileName, "r") as file:
        line = file.__next__()
        shape = tuple(map(np.int32, line.split()))
        array = np.empty(shape, dtype=dtype)
        for i, line in enumerate(file):
            array.flat[i] = dtype(line.strip())
    return array


def saveSparseArray(fileName, array):
    """
    saves a sparse matrix as a Matrix Market Coordinate file

    args:
        fileName (str): name of file to write
        array (scipy.sparse matrix): array to be saved
    """
    if not sparse.issparse(array):
        raise ValueError("array is not a scipy.sparse matrix")
    # quad matrices must be recorded manually
    if array.dtype == np.quad:
        array = array.tocoo()
        fArray = array.astype(np.float64)
        symmetry = (
            "general"
            if array.shape[0] != array.shape[1]
            else "symmetric"
            if not (fArray != fArray.transpose()).sum()
            else "skew-symmetric"
            if not (fArray != -fArray.transpose()).sum()
            else "general"
        )
        if symmetry == "symmetric" or symmetry == "skew-symmetric":
            array = sparse.tril(array)
        with open(fileName, "w") as file:
            file.write("%%MatrixMarket matrix coordinate real {}\n%\n".format(symmetry))
            # Only the lower triangle of symmetric and skew-symmetric matrices
            # are stored
            file.write("{} {} {}\n".format(array.shape[0], array.shape[1], array.nnz))
            for i, j, value in zip(array.row, array.col, array.data):
                file.write("{}\t{}\t{}\n".format(i + 1, j + 1, str(value)))
    # Boolean matrices disrupt symmetry checking, so they are coerced to ints
    elif array.dtype == np.bool:
        array = array.astype(np.int8)
        io.mmwrite(fileName, array, field="pattern")
    else:
        io.mmwrite(fileName, array)


def loadSparseArray(fileName, dtype):
    """
    loads a Matrix Market sparse matrix file onto a scipy.sparse matrix of the
    chosen type

    args:
        fileName (str): name of file to read
        dtype (type): type of output, constructor must accept strings
    """
    # quad matrices must be handled manually
    if dtype == np.quad:
        with open(fileName, "r") as file:
            # Scans first line
            line = file.__next__()
            assert line.lower().split()[:4] == [
                "%%matrixmarket",
                "matrix",
                "coordinate",
                "real",
            ], "invalid MatrixMarket description for quad matrix"
            symmetry = line.split()[4].lower()
            sym = (
                1
                if symmetry == "symmetric"
                else -1
                if symmetry == "skew-symmetric"
                else 0
            )
            # Skips through comments to shape
            for line in file:
                if line[0] != "%":
                    break
            shape0, shape1, nnz = map(np.int32, line.split())
            row = np.empty(2 * nnz if sym else nnz, dtype=np.int32)
            col = np.empty(2 * nnz if sym else nnz, dtype=np.int32)
            data = np.empty(2 * nnz if sym else nnz, dtype=np.quad)
            i = 0
            for line in file:
                entries = line.split()
                row[i] = np.int32(entries[0]) - 1
                col[i] = np.int32(entries[1]) - 1
                data[i] = np.quad(entries[2])
                i += 1
                if sym and row[i - 1] != col[i - 1]:
                    row[i], col[i] = col[i - 1], row[i - 1]
                    data[i] = sym * data[i - 1]
                    i += 1
            row.resize(i)
            col.resize(i)
            data.resize(i)
            return sparse.coo_matrix(
                (data, (row, col)), shape=(shape0, shape1), dtype=np.quad
            )
    elif dtype == np.bool:
        return io.mmread(fileName).astype(np.bool)
    else:
        return io.mmread(fileName)


def saveScalar(packingFile, scalarString, scalar):
    """
    Append to the scalars.dat file of a packing

    args:
    packingFile: The filename for your packing
    scalarString: The name of the variable
    scalar: The scalar you wish to save
    """
    scalarsFile = packingFile + "/scalars.dat"
    with open(scalarsFile, "a+") as sf:
        sf.write(scalarString + "\t" + str(scalar) + "\n")


def loadScalar(packingFile, scalarString):
    """
    Loads the first scalar labeled by scalarString from the scalars.dat in the
    packingFile.  If not found, return None. (Maybe should raise an exception?)

    args:
    packingFile: The directory for the packing
    scalarString: The name of the variable

    returns:
    scalarValue: The value of the named scalar, as a string
    """

    scalarValue = None
    with open(packingFile + "/scalars.dat", "r") as file:
        for line in file:
            name, value = line.strip().split("\t")
            if name == scalarString:
                scalarValue = value
                break
    return scalarValue


# Old file formats ------------------------------------------------------------
def readWhiteSpaceFile(fileName):
    """
    Reads a whitespace separated file onto a 2D python array of strings

    args:
        fileName (str): Path to input file
    """
    with open(fileName) as f:
        return [line.split() for line in f.readlines()]
