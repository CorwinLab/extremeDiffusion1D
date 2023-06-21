import numpy as np 
from matplotlib import pyplot as plt
from pyDiffusion.pyscattering import iteratePDFFields

def evolveToTime(right, left, N, t, xvals, rc):
	quantiles = []
	for t in range(t):
		right, left, pos = iteratePDFFields(right, left, 1/N, xvals, rc)
		quantiles.append(pos)
	return right, left, quantiles

if __name__ == '__main__':
	nSystems = 500
	size = 1000
	N = 1e5
	rc = 2
	
	quantiles = []
	for n in range(nSystems):
		right = np.zeros(size+1)
		left = np.zeros(size+1)
		right[right.size // 2] = 1
		xvals = np.arange(-size//2, size//2 + 1, 1)
		
		right, left, pos = evolveToTime(right, left, N, size//2, xvals, rc)
		quantiles.append(np.array(pos))
		print(n)

	quantiles = np.array(quantiles)
	var = []
	mean = []
	for col in range(quantiles.shape[1]):
		var.append(np.var(quantiles[:, col]))
		mean.append(np.mean(quantiles[:, col]))
	
	np.savetxt(f"./fieldsData/Var{rc}.txt", var)
	np.savetxt(f"./fieldsData/Mean{rc}.txt", mean)

	fig, ax = plt.subplots()
	ax.set_xscale("log")
	ax.set_yscale("log")
	ax.plot(np.arange(1, size//2+1), mean)
	fig.savefig("Mean.pdf", bbox_inches='tight')

	fig, ax = plt.subplots()
	ax.set_xscale("log")
	ax.set_yscale("log")
	ax.plot(np.arange(1, size // 2 + 1), var)
	fig.savefig("Var.pdf", bbox_inches='tight')