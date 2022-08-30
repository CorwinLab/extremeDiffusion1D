import numpy as np
from experimentUtils import saveVars
from pyDiffusion import pydiffusion2D
import sys
import os
from datetime import date

def runExperiment(nParticles, minTime, maxTime, num_save_times, xi, save_file, save_positions):
    save_times = np.geomspace(minTime, maxTime, num_save_times)
    pydiffusion2D.evolveAndSaveMaxDistance(nParticles, save_times, xi, save_file, save_positions)

if __name__ == '__main__':
    (
        topDir,
        sysID,
        minTime,
        maxTime,
        nParticles,
        num_save_times,
        xi,
    ) = sys.argv[1:]

    save_dir = f"{topDir}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_file = os.path.join(save_dir, f"MaxPositions{sysID}.txt")
    save_file = os.path.abspath(save_file)
    save_positions = os.path.join(save_dir, f"ParticlePositions{sysID}.txt")

    vars = {
        "nParticles": nParticles,
        "minTime": minTime,
        "maxTime": maxTime,
        "num_save_times": num_save_times,
        "xi": xi,
        "save_file": save_file,
        "save_positions": save_positions
    }

    vars_file = os.path.join(save_dir, "variables.json")
    today = date.today()
    text_date = today.strftime("%b-%d-%Y")

    if int(sysID) == 0:
        vars.update({"Date": text_date})
        saveVars(vars, vars_file)
        vars.pop("Date")

    runExperiment(**vars)
