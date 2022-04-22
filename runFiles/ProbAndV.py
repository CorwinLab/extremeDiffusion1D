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
import quadMath

def runExperiment(
    beta,
    times,
    quantile,
    save_file,
    id=None,
    save_dir="."
):
    """
    Run a simulation to get the probability and velocity at a specific quantile.

    Parameters
    ----------
    beta : float
        Value of beta for the beta distribution to draw from

    times : list
        Specific times to measure the probability and velocities at

    save_file : str
        File to save the probabilities and velocities to

    id : int
        The slurm ID used to save meta data about the system

    save_dir : str
        Path to directory of where to store meta data to.
    """
    quantile = np.quad(f"1e{quantile}")

    cdf = DiffusionTimeCDF(beta, max(times))
    cdf.id = id
    cdf.save_dir = save_dir
    cdf.evolveAndGetProbAndV(quantile, times, save_file)


if __name__ == '__main__':
    (
        topDir,
        sysID,
        beta,
        quantile,
    ) = sys.argv[1:]

    save_dir = f"{topDir}"
    save_file = os.path.join(save_dir, f"Probs{sysID}.txt")
    save_file = os.path.abspath(save_file)

    beta = float(beta)
    times = np.array([0.1, 1, 10, 100])
    times = times * np.log(np.quad(f"1e{quantile}")).astype(float)
    times = times.astype(int)
    times = [int(t) for t in times]
    sysID = int(sysID)

    vars = {
        "beta": beta,
        "times": times,
        "quantile": quantile,
        "save_file": save_file,
        "id": sysID,
        "save_dir": topDir,
    }

    vars_file = os.path.join(save_dir, "variables.json")
    today = date.today()
    text_date = today.strftime("%b-%d-%Y")

    if int(sysID) == 0:
        vars.update({"Date": text_date})
        saveVars(vars, vars_file)
        vars.pop("Date")

    runExperiment(**vars)
