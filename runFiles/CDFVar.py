import sys

sys.path.append("../src/")
import numpy as np
import npquad
import quadMath
import fileIO
import os
from experimentUtils import saveVars
from datetime import date
from pydiffusionCDF import DiffusionTimeCDF


def runExperiment(
    beta,
    tMax,
    save_file,
    num_of_save_times=5000,
    nParticles=int(1e5)
):
    """
    Run a simulation of the recurrsion relation from Ivan's model.

    Parameters
    ----------
    beta : float
        Value of beta for beta distribution to draw from.

    tMax : int
        Maximum time to run out to.

    save_file : str
        File to save the quartiles to.

    num_of_save_times : int, optional (5000)
        Number of times to save the quartiles.

    nParticles: str
        Number of particles to measure the variance of.
    """

    beta = float(beta)
    tMax = int(tMax)
    num_of_save_times = int(num_of_save_times)
    nParticles = np.quad(nParticles)

    rec = DiffusionTimeCDF(beta, tMax)

    save_times = np.geomspace(1, tMax, num_of_save_times, dtype=np.int64)
    save_times = np.unique(save_times)

    rec.evolveAndGetVariance(save_times, nParticles, save_file)

if __name__ == "__main__":
    (
        topDir,
        sysID,
        beta,
        tMax,
        num_of_save_times,
        nParticles,
    ) = sys.argv[1:]

    save_dir = f"{topDir}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_file = os.path.join(save_dir, f"Quartiles{sysID}.txt")
    save_file = os.path.abspath(save_file)

    vars = {
        "beta": beta,
        "tMax": tMax,
        "save_file": save_file,
        "num_of_save_times": num_of_save_times,
        "nParticles": nParticles,
    }
    vars_file = os.path.join(save_dir, "variables.json")
    today = date.today()
    text_date = today.strftime("%b-%d-%Y")

    runExperiment(**vars)
    vars.update({"Date": text_date})
    saveVars(vars, vars_file)
