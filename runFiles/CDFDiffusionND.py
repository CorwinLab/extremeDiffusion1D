from pyDiffusion.pydiffusionND import DiffusionND
import numpy as np
import npquad 
import csv
import os, psutil
from experimentUtils import saveVars
from datetime import date
import sys 


def runExperiment(N, alpha, Lmin, Lmax, tMax, save_file):
    process = psutil.Process(os.getpid())
    f = open(save_file, 'a')
    writer = csv.writer(f)
    writer.writerow(['Position', 'Quantile', 'Variance'])
    f.flush()
    
    Ls = np.geomspace(Lmin, Lmax, 10).astype(int)
    Ls = np.unique(Ls)

    for L in Ls: 
        # Used to see how much memory you're using
        # print(process.memory_info().rss)
        print(L)
        d = DiffusionND(alpha, tMax, L)
        quantile, variance = d.getQuantileAndVariance(N)
        writer.writerow([L, quantile, variance])
        f.flush()
        del d # python doubles the object when copying in d=Diffusion(alpha, tMax, L)

    f.close()

if __name__ == '__main__':
    (
        topDir,
        sysID,
        alpha,
        N, 
        tMax,
        Lmin,
        Lmax, 
    ) = sys.argv[1:]

    alpha = 4*[float(alpha)]
    N = float(N)
    Lmin = int(Lmin)
    Lmax = int(Lmax)*np.log(N)
    tMax = int(tMax)
    save_file=os.path.join(topDir, f"FirstPassage{sysID}.txt")
    
    vars = {"N": N,
            "alpha": alpha,
            "Lmin": Lmin,
            "Lmax": Lmax,
            "tMax": tMax,
            "save_file": save_file}
    
    vars_file = os.path.join(topDir, "variables.json")
    today = date.today()
    text_date = today.strftime("%b-%d-%Y")

    if int(sysID) == 0:
        vars.update({"Date": text_date})
        saveVars(vars, vars_file)
        vars.pop("Date")
        
    runExperiment(**vars)