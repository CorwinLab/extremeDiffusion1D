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
from sys import exit


def runExperiment(
    beta,
    N_exp,
    num_of_save_distances,
    save_file,
    save_occ,
    sysID,
    probDistFlag,
    max_distance,
    tMax,
):
    """
    Run simulation to get first passage time for some distances.
    """
    N = np.quad(f"1e{N_exp}")
    beta = float(beta)
    tMax = int(tMax)
    num_of_save_distances = int(num_of_save_distances)
    probDistFlag = bool(int(probDistFlag))
    max_distance = int(max_distance)

    logN = np.log(N).astype(float)
    distances = np.geomspace(1, max_distance, num_of_save_distances, dtype=np.int64)
    distances = np.unique(distances)

    occupancy_file = os.path.join(save_dir, f"Occupancy{sysID}.txt")
    scalars_file = os.path.join(save_dir, f"Scalars{sysID}.json")

    # if os.path.exists(occupancy_file) and os.path.exists(scalars_file):
    #    d = DiffusionPDF.fromFiles(scalars_file, occupancy_file)
    #    save_times = save_times[save_times > d.currentTime]
    #    append = True
    # else:
    d = DiffusionPDF(N, beta=beta, occupancySize=tMax, ProbDistFlag=probDistFlag)

    d.save_dir = save_dir
    d.id = sysID
    append = False

    d.evolveAndSaveFirstPassage(distances, save_file)

    fileIO.saveArrayQuad(save_occ, d.occupancy)


if __name__ == "__main__":
    (
        topDir,
        sysID,
        beta,
        N_exp,
        num_of_save_distances,
        probDistFlag,
        tMax,
    ) = sys.argv[1:]

    save_dir = f"{topDir}"
    save_file = os.path.join(save_dir, f"Quartiles{sysID}.txt")
    save_file = os.path.abspath(save_file)

    save_occ = os.path.join(save_dir, f"FinalOccupancy{sysID}.txt")
    save_occ = os.path.abspath(save_occ)
    if os.path.exists(save_occ):
        exit()
    max_distance = 100 * np.log(float(f"1e{N_exp}"))

    vars = {
        "beta": beta,
        "N_exp": N_exp,
        "num_of_save_distances": num_of_save_distances,
        "save_file": save_file,
        "probDistFlag": probDistFlag,
        "save_file": save_file,
        "sysID": sysID,
        "save_occ": save_occ,
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

    runExperiment(**vars)
