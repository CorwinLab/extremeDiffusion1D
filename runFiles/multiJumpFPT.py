from pyDiffusion.pymultijumpRW import evolveAndMeasureFPT
import sys 
import os 
from datetime import date
from experimentUtils import saveVars
import numpy as np

if __name__ == '__main__':
	# Testing code
	# (topDir, sysID, prefactor, step_size, distribution, Nexp) = '.', '0', '1e3', '5', 'notsymmetric', '5'
	
	# SSRW Code 
	# topDir = '/home/jacob/Desktop/SSRWData/'
	# sysID = '0'
	# Lmax = '2500'
	# step_size = '5'
	# distribution = 'ssrw'
	# Nexp = '12'

	(topDir, sysID, prefactor, step_size, distribution, Nexp) = sys.argv[1:]
	
	save_file = os.path.join(topDir, f"Quantiles{sysID}.txt")
	step_size = int(step_size)
	distribution = str(distribution)
	N = float(f"1e{Nexp}")
	prefactor = float(prefactor)
	
	# Calculate maximum position to go to
	width = step_size // 2
	sigma = np.sqrt(1/3 * width * (width + 1))
	beta = width / 6

	Lmax = (prefactor * sigma*4 * (sigma**2 - beta) * np.log(N)**(5/2) / beta) ** (1/3)
	Lmax = int(Lmax)

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