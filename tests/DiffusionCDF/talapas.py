import numpy as np
import npquad 
import sys 
sys.path.append("../../src")
from pydiffusionCDF import DiffusionTimeCDF

if __name__ == '__main__': 
	d = DiffusionTimeCDF(1, 10)
	d.iterateTimeStep()
	d.iterateTimeStep()
	d.iterateTimeStep()
	d.iterateTimeStep()
	d.iterateTimeStep()
	d.iterateTimeStep()
	#d.iterateTimeStep()
	d.iterateTimeStep()
	d.iterateTimeStep()
	diff = d.findQuantiles([np.quad("1e300")])
	print(d.getGumbelVariance([np.quad("1e300")]))
