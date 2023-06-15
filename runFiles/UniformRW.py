import sys 
from experimentUtils import saveVars
from pyuniformDistRW import evolveAndMeasureQuantileVelocity
import os
from datetime import date 

if __name__ == '__main__':
	# Testing line of code
	# (tMax, max_step_size, Nexp, topDir, sysID) = '1000', '10', '5', '.', '0'
	(tMax, max_step_size, Nexp, topDir, sysID) = sys.argv[1:]

	save_file = os.path.join(topDir, f"Quantiles{sysID}.txt")
	N = float(f"1e{Nexp}")

	vars = {'tMax': int(tMax),
	 		'max_step_size': int(max_step_size),
			'N': N,
			'v': [0.0001, 0.001, 0.01, 0.1, 1],
			'save_file': save_file}
	
	vars_file = os.path.join(topDir, "variables.json")
	today = date.today()
	text_date = today.strftime("%b-%d-%Y")

	if int(sysID) == 0:
		vars.update({"Date": text_date})
		saveVars(vars, vars_file)
		vars.pop("Date")

	evolveAndMeasureQuantileVelocity(**vars)