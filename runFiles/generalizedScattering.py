from pyDiffusion.pyscattering import evolveAndGetQuantileGeneralized
import numpy as np
import sys
import os 
from datetime import date
from experimentUtils import saveVars

if __name__ == '__main__':
    # For testing purposes 
    #save_dir, sysID, beta = '.', '0', '1'
    (save_dir, sysID) = sys.argv[1:]
    save_file = os.path.join(save_dir, f'Quantile{sysID}.txt')
    times = np.geomspace(1, 1e5, 10000).astype(int)
    times = np.unique(times)
    N = 1e5

    vars = {'times': times, 
            'N': N,
            'size': 4 * max(times),  
            'save_file': save_file,
            }

    # Save variables of experiment
    vars_file = os.path.join(save_dir, "variables.json")
    today = date.today()
    text_date = today.strftime("%b-%d-%Y")

    if int(sysID) == 0:
        vars.update({"Date": text_date})

        # numpy arrays aren't serializable 
        vars.update({'times': [int(i) for i in vars['times']]})
        vars.update({'size': int(vars['size'])})
        saveVars(vars, vars_file)
        vars.pop("Date")
        
        # convert times back to numpy array
        vars.update({'times': np.array(vars['times'])})

    quantiles = evolveAndGetQuantileGeneralized(**vars)