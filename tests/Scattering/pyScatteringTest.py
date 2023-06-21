from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib
import copy 
import numpy as np 
from pyDiffusion.pyscattering import iteratePDF
from scipy.special import binom

if __name__ == '__main__':
	size=1000
	right = np.zeros(size+1)
	left = np.zeros(size+1)

	right[right.size // 2] = 1
	
	maxTime = 500
	img = np.zeros(shape=(maxTime+1, right.size))
	img[0, :] = right - left
	img_pdf = np.zeros(shape=(maxTime+1, right.size))
	img_pdf[0, :] = right + left
	N = 1e10
	for i in range(maxTime):
		right, left, quantile = iteratePDF(right, left, N, dist='beta', params=np.inf)
		diff = right - left
		pdf = right + left
		img[i+1, :] = diff
		img_pdf[i+1, :] = pdf

	cmap = copy.copy(matplotlib.cm.get_cmap("seismic"))
	vmax = 1
	fig, ax = plt.subplots()
	im = ax.imshow(img.T[300:700], norm=colors.SymLogNorm(linthresh=10**-7, vmin=-1, vmax=1, base=10), cmap=cmap, interpolation='none')
	cbar = fig.colorbar(im, orientation='vertical')
	cbar.ax.set_ylabel(r"$p_{R}(x,t) - p_{L}(x,t)$")
	ax.set_ylabel("Distance")
	ax.set_xlabel("Time")
	fig.savefig("DeltaDistribution.pdf", bbox_inches='tight')

	pdf = img[-1, :][::2]
	x = np.arange(-500, 501, 2)

	def gauss_derv(x, mean, var):
		return - x / np.sqrt(2 * np.pi) / var ** (3/2) * np.exp(-(x-mean)**2 / 2 / var)
	
	def gauss(x, mean, var):
		return 1 / np.sqrt(2 * np.pi * var) * np.exp(- (x-mean)**2 / 2 / var)
	 
	def rand_binom(x, t):
		return 2**(-t) * binom(t, (t + x)/2)

	fig, ax = plt.subplots()
	ax.scatter(x, pdf, s=1)
	ax.plot(x, x / maxTime * rand_binom(x, maxTime), c='k', alpha=0.5)
	ax.set_xlim([-500, 500])
	ax.set_xlabel("x")
	ax.set_ylabel(r"$p_{R}(x,t) - p_{L}(x,t)$")
	fig.savefig("DeltaDist.pdf")
	
	fig, ax = plt.subplots()
	ax.scatter(x, img_pdf[-1, :][::2], s=1)
	ax.plot(x, rand_binom(x, maxTime), ls='--', c='k', alpha=0.75)
	ax.plot(x, 2*gauss(x, 0, maxTime), ls='-.', c='r', alpha=0.5)
	ax.set_xlim([-500, 500])
	ax.set_xlabel("x")
	ax.set_ylabel(r"$p_{\bf{B}}(x,t)$")
	fig.savefig("ProbDist.pdf")