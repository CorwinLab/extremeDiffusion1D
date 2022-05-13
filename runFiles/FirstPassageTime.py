import sys
import os

src_path = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.append(src_path)

from pydiffusionSystems import AllSystems
import numpy as np
from datetime import date
from experimentUtils import saveVars
from sys import exit
import time

def runExperiment(beta, N_exp, numSystems, num_of_save_distances, save_file, max_distance, tMax):
    """
    Run simulation to get first passage time for some distances.
    """
    N = float(f"1e{N_exp}")
    beta = float(beta)
    tMax = int(tMax)
    numSystems = int(numSystems)
    num_of_save_distances = int(num_of_save_distances)
    max_distance = int(max_distance)

    distances = np.geomspace(1, max_distance, num_of_save_distances, dtype=np.int64)
    distances = np.unique(distances)
    
    d = AllSystems(
        numSystems, beta, tMax, N
    )

    np.savetxt(save_file, d.measureFirstPassageTimes(distances))

if __name__ == "__main__":
    (
        topDir,
        sysID,
        beta,
        N_exp,
        num_of_save_distances,
        numSystems,
        tMax,
    ) = sys.argv[1:]

    save_dir = f"{topDir}"
    save_file = os.path.join(save_dir, f"FirstPassageTimes{sysID}.txt")

    max_distance = 2000
    
    vars = {
        "beta": beta,
        "N_exp": N_exp,
        "numSystems": numSystems,
        "save_file": save_file,
        "num_of_save_distances": num_of_save_distances,
        "max_distance": max_distance,
        "tMax": tMax,
    }
    vars_file = os.path.join(save_dir, "variables.json")
    today = date.today()
    text_date = today.strftime("%b-%d-%Y")

    if int(sysID) == 0:
        vars.update({"Date": text_date})
        saveVars(vars, vars_file)
        vars.pop("Date")
    s = time.time()
    runExperiment(**vars)
    np.savetxt(os.path.join(topDir, f"Time{sysID}.txt"), [time.time() - s])
