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
    nParticles,
    num_of_save_times=5000,
    sysID=None,
    save_dir=".",
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

    nParticles: list of np.quads
        Number of particles to measure the variance of.

    num_of_save_times : int, optional (5000)
        Number of times to save the quartiles.
    """

    beta = float(beta)
    tMax = int(tMax)
    num_of_save_times = int(num_of_save_times)

    save_times = np.geomspace(1, tMax, num_of_save_times, dtype=np.int64)
    save_times = np.unique(save_times)

    scalars_file = os.path.join(save_dir, f"Scalars{sysID}.json")
    CDF_file = os.path.join(save_dir, f"CDF{sysID}.txt")

    if os.path.exists(scalars_file) and os.path.exists(CDF_file):
        rec = DiffusionTimeCDF.fromFiles(CDF_file, scalars_file)
        save_times = savetimes[savetimes > rec.time]
        append = True
    else:
        rec = DiffusionTimeCDF(beta, tMax)
        rec.id = sysID
        rec.save_dir = save_dir
        append = False

    rec.evolveAndGetVariance(save_times, nParticles, save_file, append=append)


if __name__ == "__main__":
    (
        topDir,
        sysID,
        beta,
        tMax,
        num_of_save_times,
        nStart,
        nStop,
        nStep,
    ) = sys.argv[1:]

    save_dir = f"{topDir}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_file = os.path.join(save_dir, f"Quartiles{sysID}.txt")
    save_file = os.path.abspath(save_file)

    nStart = float(nStart)
    nStop = float(nStop)
    nStep = float(nStep)
    nParticles = quadMath.logarange(nStart, nStop, nStep, endpoint=True)

    vars = {
        "beta": beta,
        "tMax": tMax,
        "save_file": save_file,
        "num_of_save_times": num_of_save_times,
        "nParticles": nParticles,
        "sysID": sysID,
        "save_dir": topDir,
    }
    vars_file = os.path.join(save_dir, "variables.json")
    today = date.today()
    text_date = today.strftime("%b-%d-%Y")

    runExperiment(**vars)
    vars.update({"Date": text_date})
    saveVars(vars, vars_file)
