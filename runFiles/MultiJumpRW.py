import numpy as np
import os 
import sys 
from pyDiffusion.pymultijumpRW import evolveAndMeasureEnvAndMax
from datetime import date
from experimentUtils import saveVars

if __name__ == '__main__':
	# Testing code
	# (tMax, step_size, Nexp, topDir, sysID, distribution, params) = ('5000', '7', '28', '.', '0', 'symmetric', '')
	(tMax, step_size, Nexp, topDir, sysID, distribution, params) = sys.argv[1:]
	
	N = float(f"1e{Nexp}")
	save_file = os.path.join(topDir, f"Quantiles{sysID}.txt")
	if distribution == 'dirichlet' or distribution=='symmetric':
		params = params.split(",")
		params = np.array(params).astype(float)
	else: 
		params = np.array([])
	# evolveAndMeasureEnvAndMax(tMax, step_size, N, save_file, distribution)
	vars = {"tMax": float(tMax), 
	 		"step_size": int(step_size),
			"N": N,
			"save_file": save_file,
			"distribution": distribution,
			"params": params}
	
	vars_file = os.path.join(topDir, "variables.json")
	today = date.today()
	text_date = today.strftime("%b-%d-%Y")
	
	if int(sysID) == 0:
		vars.update({"Date": text_date})
		saveVars(vars, vars_file)
		vars.pop("Date")
	
	vars['params'] = np.array(vars['params'])
	evolveAndMeasureEnvAndMax(**vars)
