import numpy as np
import npquad
from pyDiffusion import DiffusionTimeCDF
import sys
import os
from experimentUtils import saveVars

if __name__ == "__main__":
    (topDir, sysID, minDistance, maxDistance, num_save_points, Nexp) = sys.argv[1:]

    minDistance = int(minDistance)
    maxDistance = int(maxDistance)
    num_save_points = int(num_save_points)

    save_file = os.path.join(topDir, f"FPT{sysID}.txt")

    N = float(f"1e{Nexp}")
    maxTime = int((maxDistance**2 / 2 / np.log(N)) * 2)
    distances = np.geomspace(minDistance, maxDistance, num_save_points)

    if sysID == 0:
        var_save_file = os.path.join(topDir, "variables.json")
        vars = {"distances": distances, "N": N, "save_file": save_file}
        saveVars(vars, var_save_file)

    cdf = DiffusionTimeCDF("beta", [1, 1], maxTime)
    cdf.evolveAndSaveFirstPassageDoubleSided(N, distances, save_file)
