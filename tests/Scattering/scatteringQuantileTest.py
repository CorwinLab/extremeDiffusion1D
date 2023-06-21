from pyDiffusion.pyscattering import iteratePDF
from matplotlib import pyplot as plt
import numpy as np
from scipy.special import binom

def rand_binom(x, t):
    return 2**(-t) * binom(t, (t + x)/2)

if __name__ == '__main__': 
    size=2000
    right = np.zeros(size+1)
    left = np.zeros(size+1)
    xvals = np.arange(-size//2, size//2+1, 1)
    N = 1e10

    rand_initial = np.random.uniform(0, 1)
    right[right.size // 2] = rand_initial
    left[left.size // 2] = 1-rand_initial
    
    for i in range(size//2-1):
        right, left, quantile = iteratePDF(right, left, 1/N)
        pdf = right+left 
        diff = right - left
        if i % 2 == 1:
            fig, ax = plt.subplots()
            ax.plot(xvals[::2], pdf[::2])
            ax.vlines(quantile, 10**-20, 1, ls='--', color='r')
            ax.plot(xvals[size//2 - i : size//2 + i+1:2], rand_binom(xvals[size//2 - i : size//2 + i+1:2], i), ls='--', c='k', alpha=0.75)
            ax.set_yscale("log")
            ax.set_ylim([10**-20, 1])
            ax.set_xlim([-300, 300])
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$p_{\bf{B}}(x)$")
            ax.set_title(f"t={i}")
            fig.savefig(f"./PDFs/PDF{i}.png")
            plt.close()
            
            fig, ax = plt.subplots()
            ax.plot(xvals[::2], diff[::2])
            ax.plot(xvals[size//2 - i : size//2 + i+1:2], rand_binom(xvals[size//2 - i : size//2 + i+1:2], i), ls='--', c='k', alpha=0.75)
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$p_{\bf{B}, R}(x) - p_{\bf{B}, L}(x)$")
            ax.set_title(f"t={i}")
            ax.set_yscale("log")
            ax.set_xlim([0, 300])
            ax.set_ylim([10**-20, 1])
            ax.vlines(quantile, 10**-20, 1, ls='--', color='r')
            fig.savefig(f"./Diffs/PDF{i}.png")
            plt.close()