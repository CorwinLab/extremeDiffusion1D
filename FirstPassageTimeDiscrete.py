from pyDiffusion import DiffusionPDF, quadMath, fileIO
import numpy as np
import npquad
from datetime import date
from sys import exit
import sys
import os
from multiprocessing import Pool, cpu_count
sys.path.append("./runFiles")
from experimentUtils import saveVars

def runExperiment(
    sysID,
):
    """
    Run simulation to get first passage time for some distances.
    """
    save_dir = "./DiscreteSSRW"
    N_exp = 12
    beta = np.inf
    num_of_save_distances = 7500
    probDistFlag = 0
    tMax = 13_000_000
    max_distance = 1000 * np.log(float(f"1e{N_exp}"))

    save_file = os.path.join(save_dir, f"Quartiles{sysID}.txt")
    save_file = os.path.abspath(save_file)

    save_occ = os.path.join(save_dir, f"FinalOccupancy{sysID}.txt")
    save_occ = os.path.abspath(save_occ)
    if os.path.exists(save_occ):
        print("Found FinalOccupancy file so exiting", flush=True)
        exit()

    vars_file = os.path.join(save_dir, "variables.json")
    today = date.today()
    text_date = today.strftime("%b-%d-%Y")
    
    vars = {
        "beta": beta,
        "N_exp": N_exp,
        "num_of_save_distances": num_of_save_distances,
        "probDistFlag": probDistFlag,
        "sysID": sysID,
        "max_distance": max_distance,
        "tMax": tMax,
        "save_dir": save_dir
    }

    if int(sysID) == 0:
        vars.update({"Date": text_date})
        saveVars(vars, vars_file)
        vars.pop("Date")

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

    if os.path.exists(occupancy_file) and os.path.exists(scalars_file):
        d = DiffusionPDF.fromFiles(scalars_file, occupancy_file)
        quartiles = np.loadtxt(save_file, skiprows=1, delimiter=',')
        current_distance = quartiles[-1, 0]
        distances = distances[distances > current_distance]
        append = True
    else:
        d = DiffusionPDF(N, 'beta', parameters=[beta, beta], occupancySize=tMax, ProbDistFlag=probDistFlag)
        d.save_dir = save_dir
        d.id = sysID
        append = False

    if os.path.exists(save_file):
        quartiles = np.loadtxt(save_file, skiprows=1, delimiter=',')
        current_distance = quartiles[-1, 0]
        distances = distances[distances > current_distance]
        append = True
        
    if d.getOccupancySize() < tMax:
        d.resizeOccupancy(tMax)

    d.evolveAndSaveFirstPassage(distances, save_file, append)

    fileIO.saveArrayQuad(save_occ, d.occupancy)


if __name__ == "__main__":
    ids = list(range(0, 500))
    with Pool(cpu_count()-1) as p:
        p.map(runExperiment, ids)