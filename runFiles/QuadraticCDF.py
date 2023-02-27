import numpy as np
import npquad
import os
from experimentUtils import saveVars
from datetime import date
import sys
from pyDiffusion import DiffusionTimeCDF

def runExperiment(
    a,
    b,
    tMax,
    save_file,
    n_exp,
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

    tMax = int(tMax)
    num_of_save_times = int(num_of_save_times)
    
    nParticles = [np.quad(f"1e{i}") for i in n_exp]
    
    save_times = np.geomspace(1, tMax, num_of_save_times, dtype=np.int64)
    save_times = np.unique(save_times)

    scalars_file = os.path.join(save_dir, f"Scalars{sysID}.json")
    CDF_file = os.path.join(save_dir, f"CDF{sysID}.txt")

    if os.path.exists(scalars_file) and os.path.exists(CDF_file):
        rec = DiffusionTimeCDF.fromFiles(CDF_file, scalars_file)
        save_times = save_times[save_times > rec.time]
        append = True
    else:
        rec = DiffusionTimeCDF('quadratic', [a, b], tMax)
        rec.id = sysID
        rec.save_dir = save_dir
        append = False
        
    # Make sure this is actually bates distributed:
    vars = []
    for _ in range(100000):
        vars.append(rec.generateRandomVariable())
    np.savetxt(os.path.join(save_dir, f"RandomNums{sysID}.txt"), vars)

    rec.evolveAndGetVariance(save_times, nParticles, save_file, append=append)

if __name__ == "__main__":
    # Test Line
    # topDir, sysID, beta, num_of_save_times = '.', 1, 1, 10
    (
        topDir,
        sysID,
        num_of_save_times,
    ) = sys.argv[1:]
    
    save_dir = f"{topDir}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_file = os.path.join(save_dir, f"Quartiles{sysID}.txt")
    save_file = os.path.abspath(save_file)

    n_exp = [2, 7, 24, 85, 300]
    tMax = np.log(1e24) * 5 * 10**3
    a = 1/2 * (1 - np.sqrt(5/9))
    b = 1-a
    vars = {
        "a": a,
        "b": b,
        "tMax": tMax,
        "save_file": save_file,
        "num_of_save_times": num_of_save_times,
        "n_exp": n_exp,
        "sysID": sysID,
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
