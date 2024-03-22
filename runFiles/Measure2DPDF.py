import numpy as np
import os 
import sys 
sys.path.append("../examples/")
from evolve2DLattice import measurePDFBeyondRad
from datetime import date
from experimentUtils import saveVars

if __name__ == '__main__':
	# Testing code
	# (topDir, sysID, tMax) = '.', '0', '100'
	(topDir, sysID, tMax) = sys.argv[1:]
	
	save_file = os.path.join(topDir, f"PDF{sysID}.txt")
	rs = list(np.arange(10, 110, step=10).astype(float))
	tMax = int(tMax)
	# evolveAndMeasureEnvAndMax(tMax, step_size, N, save_file, distribution)
	vars = {"tMax": tMax, 
	 		"save_file": save_file,
			"rs": rs}
	
	vars_file = os.path.join(topDir, "variables.json")
	today = date.today()
	text_date = today.strftime("%b-%d-%Y")
	
	if int(sysID) == 0:
		vars.update({"Date": text_date})
		saveVars(vars, vars_file)
		vars.pop("Date")

	measurePDFBeyondRad(**vars)
