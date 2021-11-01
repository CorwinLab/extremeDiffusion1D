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
    probDistFlag=1,
    save_dir=".",
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

    probDistFlag : bool (1 or True)
        Whether or not to keep the number of particles constant or not.
    """

    N = np.quad(f"1e{N_exp}")
    beta = float(beta)
    num_of_steps = int(num_of_steps)
    num_of_save_times = int(num_of_save_times)
    probDistFlag = bool(int(probDistFlag))

    save_times = np.geomspace(1, num_of_steps, num_of_save_times, dtype=np.int64)
    save_times = np.unique(save_times)

    occupancy_file = os.path.join(self.save_dir, f"Occupancy{self.id}.txt")
    scalars_file = os.path.join(self.save_dir, f"Scalars{self.id}.json")

    if os.path.exists(occupancy_file) and os.path.exists(scalars_file):
        d = DiffusionPDF.fromFiles(scalars_file, occupancy_file)
        save_times = save_times[save_times > d.currentTime]
        append = True
    else:
        d = DiffusionPDF(
            N, beta=beta, occupancySize=num_of_steps, ProbDistFlag=probDistFlag
        )
        d.save_dir = save_dir
        d.id = sysID
        append = False

    d.evolveAndSaveMax(save_times, save_file, append=append)

    fileIO.saveArrayQuad(save_occ, d.occupancy)


if __name__ == "__main__":
    (
        topDir,
        sysID,
        beta,
        N_exp,
        num_of_steps,
        num_of_save_times,
        probDistFlag,
    ) = sys.argv[1:]

    save_dir = f"{topDir}/{N_exp}/"
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
