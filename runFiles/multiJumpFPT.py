from pyDiffusion.pymultijumpRW import evolveAndMeasureFPT
import sys 
import os 
from datetime import date
from experimentUtils import saveVars

if __name__ == '__main__':
	# Testing code
	# (topDir, sysID, Lmax, step_size, distribution, Nexp) = '.', '0', '50', '5', 'symmetric', '5'
	
	# SSRW Code 
	# topDir = '/home/jacob/Desktop/SSRWData/'
	# sysID = '0'
	# Lmax = '2500'
	# step_size = '5'
	# distribution = 'ssrw'
	# Nexp = '12'

	(topDir, sysID, Lmax, step_size, distribution, Nexp) = sys.argv[1:]
	
	save_file = os.path.join(topDir, f"Quantiles{sysID}.txt")
	Lmax = int(Lmax)
	step_size = int(step_size)
	distribution = str(distribution)
	N = float(f"1e{Nexp}")

	vars = {"Lmax": Lmax, 
	 		"step_size": step_size,
			"distribution": distribution,
			"save_file": save_file,
			"N": N}

	vars_file = os.path.join(topDir, "variables.json")
	today = date.today()
	text_date = today.strftime("%b-%d-%Y")

	if int(sysID) == 0:
		vars.update({"Date": text_date})
		saveVars(vars, vars_file)
		vars.pop("Date")

	evolveAndMeasureFPT(**vars)