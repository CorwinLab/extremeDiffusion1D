from continuousFokkerPlanck import evolveAndGetProbs
import sys 
import os
from datetime import date 
from experimentUtils import saveVars

if __name__ == '__main__':
	# Testing code
	(topDir, sysID, tMax, v, D0, sigma, dx) = '.', '0', '50000', '0.2', '0.01', '0.001', '0.05'
	#(topDir, sysID, tMax, v, D0, sigma, dx) = sys.argv[1:]
	save_file = os.path.join(topDir, f'ProbDist{sysID}.txt')

	vars = {'tMax': int(tMax),
	 		'v': float(v),
			'D0': float(D0), 
			'sigma': float(sigma),
			'dx': float(dx),
			'save_file': save_file}
	vars_file = os.path.join(topDir, "variables.json")
	today = date.today()
	text_date = today.strftime("%b-%d-%Y")

	if int(sysID) == 0:
		vars.update({"Date": text_date})
		saveVars(vars, vars_file)
		vars.pop("Date")

	evolveAndGetProbs(**vars)
