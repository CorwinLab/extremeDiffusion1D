import sys
import os

sys.path.append(os.path.abspath("../src"))
sys.path.append(os.path.abspath("../cDiffusion"))
from pydiffusion import Diffusion
import quadMath
import numpy as np
import npquad
from datetime import date
from experimentUtils import saveVars


def runExperiment(
    beta,
    N_exp,
    num_of_steps,
    save_file,
    num_of_save_times=5000,
    v_start=0.1,
    v_stop=0.9,
    v_step=0.1,
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
        What file to save the qu
        artiles to.

    num_of_save_times : int, optional (5000)
        Number of times to save the quartiles.

    v_start : float, optional (0.1)
        Starting velocity to measure

    v_stop : float, optional (0.9)
        Final (maximum) velocity to measure

    v_step : float, optional (0.1)
        Step size between velocities.
    """

    N = np.quad(f"1e{N_exp}")
    beta = float(beta)
    num_of_steps = int(num_of_steps)
    num_of_save_times = int(num_of_save_times)
    v_start = float(v_start)
    v_stop = float(v_stop)
    v_step = float(v_step)

    d = Diffusion(N, beta=beta, occupancySize=num_of_steps, probDistFlag=True)

    save_times = np.geomspace(1, num_of_steps, num_of_save_times, dtype=np.int64)
    save_times = np.unique(save_times)

    velocities = np.arange(v_start, v_stop+v_step, v_step)

    d.evolveAndSaveV(save_times, velocities, save_file)


if __name__ == "__main__":
    (
        topDir,
        sysID,
        beta,
        N_exp,
        num_of_steps,
        num_of_save_times,
        v_start,
        v_stop,
        v_step,
    ) = sys.argv[1:]

    save_dir = f"{topDir}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_file = save_dir + f"Quartiles{sysID}.txt"
    save_file = os.path.abspath(save_file)

    vars = {
        "beta": beta,
        "N_exp": N_exp,
        "num_of_steps": num_of_steps,
        "save_file": save_file,
        "num_of_save_times": num_of_save_times,
        "v_start": v_start,
        "v_stop": v_stop,
        "v_step": v_step,
    }

    vars_file = os.path.join(save_dir, "variables.json")
    today = date.today()
    text_date = today.strftime("%b-%d-%Y")

    runExperiment(**vars)
    vars.update({"Date": text_date})
    saveVars(vars, vars_file)
