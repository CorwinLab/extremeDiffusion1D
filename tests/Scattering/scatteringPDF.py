import numpy as np 
from matplotlib import pyplot as plt
from pyDiffusion.pyscattering import iteratePDF

def evolveToTime(right, left, N, t):
	quantiles = []
	for _ in range(t):
		right, left, pos = iteratePDF(right, left, 1/N)
		quantiles.append(pos)
	return right, left, quantiles

if __name__ == '__main__':
	size = 1000
	N = 1e5
	nSystems = 500
	xvals = np.arange(-size // 2, size // 2 +1, 1)

	fig, ax = plt.subplots()
	ax.set_yscale("log")
	ax.set_ylim([10**-20, 10**-1])
	ax.set_xlim([0, 500])

	fig2, ax2 = plt.subplots()
	ax2.set_yscale("log")
	ax2.set_ylim([10**-20, 10**-1])
	ax2.set_xlim([0, 500])
	
	fig3, ax3 = plt.subplots()
	ax3.set_yscale("log")
	ax3.set_ylim([10**-20, 10**-1])
	ax3.set_xlim([0, 500])
	quantiles = []
	axins = ax3.inset_axes([0.2, 0.2, 0.5, 0.5])

	for n in range(nSystems):
		rand_initial = np.random.uniform(0, 1)
		right = np.zeros(size+1)
		left = np.zeros(size+1)
		right[right.size // 2] = rand_initial
		left[left.size // 2] = 1-rand_initial
		
		right, left, pos = evolveToTime(right, left, N, size//2)
		quantiles.append(np.array(pos))
		ax.plot(xvals[::2], right[::2], alpha=0.5)
		ax2.plot(xvals[::2], left[::2], alpha=0.5)
		ax3.plot(xvals[::2], np.cumsum(right[::2] + left[::2]), alpha=0.5)
		#print(right, left)
		#axins.plot(xvals[::2], np.sum(right[::2] + left[::2]), alpha=0.5)
		print(n)
	quantiles = np.array(quantiles)
	var = []
	for col in range(quantiles.shape[1]):
		var.append(np.var(quantiles[:, col]))


	axins.set_xlim([295, 310])
	axins.set_ylim([10**-5 / 10, 0.2*10**-5])
	ax3.indicate_inset_zoom(axins, edgecolor='black')

	fig.savefig("Right.png", bbox_inches='tight')
	fig2.savefig("Left.png", bbox_inches='tight')
	fig3.savefig("PDF.png", bbox_inches='tight')

	fig, ax = plt.subplots()
	ax.set_xscale("log")
	ax.set_yscale("log")
	ax.plot(np.array(range(0, size//2)) / np.log(N), var)
	fig.savefig("Variance.png", bbox_inches='tight')