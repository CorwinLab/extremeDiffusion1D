import numpy as np 
from pyDiffusion import DiffusionTimeCDF
import os
import sys
from datetime import date
from experimentUtils import saveVars

def runExperiment(x, times, save_dir, id):
    tMax = max(times)
    cdf = DiffusionTimeCDF('beta', [1, 1], tMax+1)
    cdf.id = id

    save_file = os.path.join(save_dir, f"CDF{id}.txt")
    scalars_file = os.path.join(save_dir, f"Scalars{id}.json")
    CDF_file = os.path.join(save_dir, f"CDF{id}.txt")

    if os.path.exists(scalars_file) and os.path.exists(CDF_file):
        rec = DiffusionTimeCDF.fromFiles(CDF_file, scalars_file)

    if os.path.exists(save_file):
        data = np.loadtxt(save_file)
        times = data[:, 0]
        if max(times) == save_times[-1]:
            sys.exit()
        save_times = save_times[save_times > times]
        write_header = False

    cdf.measureFirstPassageCDF(save_times, x, save_file, write_header=write_header)

if __name__ == '__main__':
    (topDir, x, tMax, nPoints, id) = sys.argv[1:]
    times = np.unique(np.geomspace(1, int(tMax), int(nPoints)).astype(int))

    vars = {
        'x': int(x),
        'times': list(times),
        'save_dir': topDir,
        'id': int(id),
    }

    vars_file = os.path.join(topDir, "variables.json")
    today = date.today()
    text_date = today.strftime("%b-%d-%Y")

    if int(int(id)) == 0:
        vars.update({"Date": text_date})
        saveVars(vars, vars_file)
        vars.pop("Date")

    runExperiment(**vars)