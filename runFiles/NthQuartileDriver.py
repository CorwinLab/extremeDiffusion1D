import sys
import os

src_path = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.append(src_path)

from pydiffusionPDF import DiffusionPDF
import quadMath
import fileIO
import numpy as np
import npquad
from datetime import date
from experimentUtils import saveVars


def runExperiment(
    beta,
    N_exp,
    num_of_steps,
    save_file,
    save_occ,
    num_of_save_times=5000,
    q_start=50,
    q_stop=4500,
    q_step=50,
    probDistFlag=1,
    save_dir='.',
    sysID=None,
):
    """
    Run one Diffusion experiment for values of N & beta and then store the edges
    in filename.

    Parameters
    ----------
    beta : float
        Value of beta for beta distribution

    N_exp : int
        Exponent of the number of particles to simulate. Number of particles will
        be 10 ** N_exp

    num_of_steps : int
        Number of times to evolve the system or maximum time.

    save_file : str
        What file to save the quartiles to.

    save_occ : str
        What file to save the occupancy to.

    num_of_save_times : int, optional (5000)
        Number of times to save the quartiles.

    q_start : int, optional (50)
        Exponent of first quartile to measure. Quartile will be 10 ** q_start

    q_stop : int, optional (4500)
        Exponent of last quartile to measure. Quartile will be 10 ** q_stop

    q_step : int, optional (50)
        Exponent step size between quartiles.

    probDistFlag : bool (1 or True)
        Whether or not to keep the number of particles constant or not.
    """

    N = np.quad(f"1e{N_exp}")
    beta = float(beta)
    num_of_steps = int(num_of_steps)
    num_of_save_times = int(num_of_save_times)
    q_start = int(q_start)
    q_stop = int(q_stop)
    q_step = int(q_step)
    probDistFlag = bool(int(probDistFlag))

    d = DiffusionPDF(
        N, beta=beta, occupancySize=num_of_steps, ProbDistFlag=probDistFlag
    )
    d.save_dir = save_dir
    d.id = sysID
    save_times = np.geomspace(1, num_of_steps, num_of_save_times, dtype=np.int64)
    save_times = np.unique(save_times)

    # Note: the quartiles will be sorted in descending order in evolveAndSaveQuartiles
    quartiles = quadMath.logarange(q_start, q_stop, q_step, endpoint=True)

    d.evolveAndSaveQuantiles(save_times, quartiles, save_file)

    fileIO.saveArrayQuad(save_occ, d.occupancy)


if __name__ == "__main__":
    (
        topDir,
        sysID,
        beta,
        N_exp,
        num_of_steps,
        num_of_save_times,
        quartile_start,
        quartile_stop,
        q_step,
        probDistFlag,
    ) = sys.argv[1:]

    save_dir = f"{topDir}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_file = save_dir + f"Quartiles{sysID}.txt"
    save_file = os.path.abspath(save_file)

    save_occ = save_dir + f"FinalOccupancy{sysID}.txt"
    save_occ = os.path.abspath(save_occ)

    vars = {
        "beta": beta,
        "N_exp": N_exp,
        "num_of_steps": num_of_steps,
        "save_file": save_file,
        "save_occ": save_occ,
        "num_of_save_times": num_of_save_times,
        "q_start": quartile_start,
        "q_stop": quartile_stop,
        "q_step": q_step,
        "probDistFlag": probDistFlag,
        "save_dir": topDir,
        "sysID": sysID,
    }
    vars_file = os.path.join(save_dir, "variables.json")
    today = date.today()
    text_date = today.strftime("%b-%d-%Y")

    runExperiment(**vars)
    vars.update({"Date": text_date})
    saveVars(vars, vars_file)
