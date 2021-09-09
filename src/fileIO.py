import numpy as np
import npquad
import csv


def loadArrayQuad(file, shape, skiprows=0, delimiter=","):
    """
    Load a quad precision array from a file.

    Parameters
    ---------
    file : str
        Path to file

    shape : tuple
        Shape of the array to load

    skiprows : int (0)
        Number of rows to skip before reading the data

    delimiter : str (",")
        Character to split the rows on

    Returns
    -------
    arr : numpy array (dtype=np.quad)
        Data as an array with quad precision

    Throws
    ------
    ValueError
        Throws if shape is larger than the size of the file. This is to avoid
        silently returning values at the end of the array that are empty.
    """

    arr = np.empty(shape, dtype=np.quad)
    with open(file, "r") as f:
        if skiprows > 0:
            for _ in range(skiprows):
                f.readline()
        for row, line in enumerate(f):
            # strip first to get rid of "/n"
            line = line.strip().split(delimiter)
            for col, elem in enumerate(line):
                elem = np.quad(elem)
                if arr.ndim == 1:
                    arr[col] = elem
                else:
                    arr[row, col] = elem

    if arr.ndim != 1:
        if (row != shape[0] - 1) and (col != shape[1] - 1):
            raise ValueError("Data is not the same size as the shape")

    return arr


def saveArrayQuad(save_file, arr):
    """
    Save a numpy array of quads to a file.

    Parameters
    ----------
    save_file : str
        File to save array to

    arr : numpy array (dtype np.quad)
        Array to save
    """

    if isinstance(arr, list):
        arr = np.array(arr, dtype=np.quad)
        
    with open(save_file, "w+") as f:
        writer = csv.writer(f)
        if arr.ndim == 1:
            writer.writerow(arr)
        else:
            for row in range(arr.shape[0]):
                writer.writerow(arr[row, :])
