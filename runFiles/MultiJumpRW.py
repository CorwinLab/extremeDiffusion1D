import numpy as np
import os 
import sys 
from pyDiffusion.pymultijumpRW import evolveAndMeasureQuantileVelocity
from datetime import date
from experimentUtils import saveVars

if __name__ == '__main__':
	# Testing code
	#(tMax, step_size, Nexp, v, topDir, sysID) = ('1000', '11', '12', '0.5', '.', '0')
	(tMax, step_size, Nexp, v, topDir, sysID) = sys.argv[1:]
	
	N = float(f"1e{Nexp}")
	save_file = os.path.join(topDir, f"Quantiles{sysID}.txt")

	vars = {"tMax": float(tMax), 
	 		"step_size": int(step_size),
			"N": N,
			"v": float(v),
			"save_file": save_file}
	
	vars_file = os.path.join(topDir, "variables.json")
	today = date.today()
	text_date = today.strftime("%b-%d-%Y")

	if int(sysID) == 0:
		vars.update({"Date": text_date})
		saveVars(vars, vars_file)
		vars.pop("Date")

	evolveAndMeasureQuantileVelocity(**vars)