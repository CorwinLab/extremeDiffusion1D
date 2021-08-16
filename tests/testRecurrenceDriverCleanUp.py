import os
import pytest


def test_createdQuartiles_andVars():
    """
    Going to test if we saved the quartiles, occupancy and variables. The
    created files should be: 'Quartiles1.txt', 'Occupancy1.txt' and 'variables.json'.
    """

    files = ["Quartiles1.txt", "zB1.txt", "variables.json"]
    for f in files:
        assert os.path.exists(f)  # make sure the file path exists
        assert os.stat(f).st_size > 0  # make sure the files aren't empty
        os.remove(f)
