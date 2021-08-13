import numpy as np
import npquad

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

    Note
    ----
    It doesn't look like this will throw an error if the shape is incorrect.
    Should probably just get rid of the shape parameter overall and append
    to the empty numpy array. Did this to avoid np.quad errors but it's
    going to core dump either way.
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
                arr[row, col] = elem

    if (row != shape[0] - 1) and (col != shape[1] - 1):
        raise ValueError("Data is not the same size as the shape")

    return arr
