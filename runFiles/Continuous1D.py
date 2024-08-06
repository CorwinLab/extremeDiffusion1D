import numpy as np
import sys 
import os 
from pyDiffusion.pycontinuous1D import evolveAndSave
from experimentUtils import saveVars
from datetime import date

if __name__ == '__main__': 
	# Testing Code
	(topDir, sysID, Nexp, tMax, xi, sigma, tol, D) = '.', '0', '5', '10000', '1', '1', '0.01', '5'
	
	# (topDir, sysID, Nexp, tMax, xi, sigma, tol, D) = sys.argv[1:]
	
	save_file = os.path.join(topDir, f"Positions{sysID}.txt")

	vars = {"tMax": int(float(tMax)), 
		 	"N": int(float(f"1e{Nexp}")),
			"xi": float(xi),
			"sigma": float(sigma),
			"tol": float(tol),
			"D": float(D),
			"save_file": save_file}

	vars_file = os.path.join(topDir, "variables.json")
	today = date.today()
	text_date = today.strftime("%b-%d-%Y")

	if int(sysID) == 0:
		vars.update({"Date": text_date})
		saveVars(vars, vars_file)
		vars.pop("Date")

	evolveAndSave(**vars)