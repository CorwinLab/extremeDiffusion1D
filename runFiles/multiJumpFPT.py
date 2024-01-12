from pyDiffusion.pymultijumpRW import evolveAndMeasureFPT, getSigmaBetaDirichlet
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

	(topDir, sysID, prefactor, step_size, distribution, Nexp, params) = sys.argv[1:]
	
	save_file = os.path.join(topDir, f"Quantiles{sysID}.txt")
	step_size = int(step_size)
	distribution = str(distribution)
	N = float(f"1e{Nexp}")
	prefactor = float(prefactor)
	
	if distribution == 'dirichlet':
		params = params.split(",")
		params = np.array(params).astype(float)
	else:
		params = np.array([])

	# Calculate maximum position to go to
	if distribution == 'uniform':
		width = step_size // 2
		sigma = np.sqrt(1/3 * width * (width + 1))
		beta = width / 6
	elif distribution == 'rwre':
		sigma = 1
		beta = 1/3
	elif distribution == 'delta':
		width = step_size // 2
		sigma = np.sqrt(1/3 * width * (width + 1))
		beta = (2 * width -1)* width *(width + 1) / 12 / width
	elif distribution == 'dirichlet':
		sigma, beta = getSigmaBetaDirichlet(params)

	Lmax = (prefactor * sigma**4 * (sigma**2 - beta) * np.log(N)**(5/2) / beta / np.sqrt(np.pi) * 2 * np.sqrt(2)) ** (1/3)
	Lmax = int(Lmax)
	
	vars = {"Lmax": Lmax, 
	 	"step_size": step_size,
		"distribution": distribution,
		"save_file": save_file,
		"N": N,
		"params": params}

	vars_file = os.path.join(topDir, "variables.json")
	today = date.today()
	text_date = today.strftime("%b-%d-%Y")

	if int(sysID) == 0:
		vars.update({"Date": text_date})
		saveVars(vars, vars_file)
		vars.pop("Date")
	vars["params"] = np.array(vars["params"])

	evolveAndMeasureFPT(**vars)
