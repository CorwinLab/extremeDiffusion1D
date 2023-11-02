import numpy as np
import sys 
from experimentUtils import saveVars
from pybinomialDistRW import evolveAndMeasureQuantileVelocity
import os
from datetime import date 

if __name__ == '__main__':
	# Testing line of code
	#(tMax, max_step_size, v, Nexp, topDir, sysID) = '1000', '10', '0.5', '5', '.', '0'
	(tMax, max_step_size, v, Nexp, topDir, sysID) = sys.argv[1:]

	save_file = os.path.join(topDir, f"Quantiles{sysID}.txt")
	N = float(f"1e{Nexp}")
	save_pdf = os.path.join(topDir, f"PDF{sysID}.txt")
	vars = {'tMax': int(tMax),
	 		'max_step_size': int(max_step_size),
			'N': N,
			'v': float(v),
			'save_file': save_file,
			'save_pdf': save_pdf}
	
	vars_file = os.path.join(topDir, "variables.json")
	today = date.today()
	text_date = today.strftime("%b-%d-%Y")

	if int(sysID) == 0:
		vars.update({"Date": text_date})
		saveVars(vars, vars_file)
		vars.pop("Date")

	evolveAndMeasureQuantileVelocity(**vars)
