import numpy as np
import os 
import sys 
from pyDiffusion.pymultijumpRW import evolveAndMeasurePDFSymmetric
from datetime import date
from experimentUtils import saveVars

if __name__ == '__main__':
	# Testing code
	# (tMax, step_size, topDir, sysID, distribution) = ('5000', '5', './Test', '0', 'betaBinom')
	(tMax, step_size, topDir, sysID, distribution) = sys.argv[1:]
	
	save_file = os.path.join(topDir, f"Quantiles{sysID}.txt")
	vs = list(np.geomspace(10, 1/1000, num=25, endpoint=True))
	alpha=11/12
	# evolveAndMeasureEnvAndMax(tMax, step_size, N, save_file, distribution)
	vars = {"tMax": float(tMax), 
	 		"step_size": int(step_size),
			"vs": vs,
			"alpha": alpha,
			"save_file": save_file,
			"distribution": distribution,
			"params": np.array([])}
	
	vars_file = os.path.join(topDir, "variables.json")
	today = date.today()
	text_date = today.strftime("%b-%d-%Y")
	
	if int(sysID) == 0:
		vars.update({"Date": text_date})
		saveVars(vars, vars_file)
		vars.pop("Date")
	
	vars['params'] = np.array(vars['params'])
	# evolveAndMeasurePDFSymmetric(tMax, step_size, vs, save_file, distribution='uniform', params=np.array([]))
	evolveAndMeasurePDFSymmetric(**vars)
