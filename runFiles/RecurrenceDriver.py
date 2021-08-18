import sys

sys.path.append("../src/")
sys.path.append("../recurrenceRelation")
import numpy as np
import npquad
import quadMath
import fileIO
import os
from experimentUtils import saveVars
from pyrecurrence import Recurrence
from datetime import date


def runExperiment(
    beta,
    tMax,
    save_file,
    save_zB,
    num_of_save_times=5000,
    q_start=50,
    q_stop=4500,
    q_step=50,
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

    save_zB : str
        What file to save zB to.

    num_of_save_times : int, optional (5000)
        Number of times to save the quartiles.

    q_start : int, optional (50)
        Exponent of first quartile to measure. Quartile will be 10 ** q_start

    q_stop : int, optional (4500)
        Exponent of last quartile to measure. Quartile will be 10 ** q_stop

    q_step : int, optional (50)
        Exponent step size between quartiles.
    """

    beta = float(beta)
    tMax = int(tMax)
    num_of_save_times = int(num_of_save_times)
    q_start = int(q_start)
    q_stop = int(q_stop)
    q_step = int(q_step)

    rec = Recurrence(beta, tMax)

    save_times = np.geomspace(1, tMax, num_of_save_times, dtype=np.int64)
    save_times = np.unique(save_times)

    quartiles = quadMath.logarange(q_start, q_stop, q_step, endpoint=True)

    rec.evolveAndSaveQuartile(save_times, quartiles, save_file)

    fileIO.saveArrayQuad(save_zB, rec.zB)


if __name__ == "__main__":
    (
        topDir,
        sysID,
        beta,
        tMax,
        num_of_save_times,
        quartile_start,
        quartile_stop,
        quartile_step,
    ) = sys.argv[1:]

    save_dir = f"{topDir}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_file = save_dir + f"Quartiles{sysID}.txt"
    save_file = os.path.abspath(save_file)

    save_zB = save_dir + f"zB{sysID}.txt"
    save_zB = os.path.abspath(save_zB)

    vars = {
        "beta": beta,
        "tMax": tMax,
        "save_file": save_file,
        "save_zB": save_zB,
        "num_of_save_times": num_of_save_times,
        "q_start": quartile_start,
        "q_stop": quartile_stop,
        "q_step": quartile_step,
    }
    vars_file = os.path.join(save_dir, "variables.json")
    today = date.today()
    text_date = today.strftime("%b-%d-%Y")

    runExperiment(**vars)
    vars.update({"Date": text_date})
    saveVars(vars, vars_file)
