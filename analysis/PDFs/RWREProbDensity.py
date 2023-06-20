# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 09:11:28 2022

@author: jacob
"""

import sys 
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 15})

def initializePDF(maxPosition):
    pdf = np.zeros(2*maxPosition + 1)
    pdf[maxPosition] = 1
    return pdf

def iteratePDF(pdf, model='RWRE'):
    if model=='RWRE':
        biases = np.random.uniform(0, 1, size=pdf.shape)
    else:
        biases = np.ones(shape=pdf.shape) / 2
        
    pdf_new = np.zeros(pdf.shape)
    for i in range(1, len(pdf)-1):
        pdf_new[i] = pdf[i-1] * biases[i-1] + pdf[i+1] * (1- biases[i+1])
    return pdf_new

if __name__ == '__main__':
    maxPosition = 500
    pdf = initializePDF(maxPosition)
    xvals = np.arange(-maxPosition, maxPosition+1, 1)
    xvals = xvals[::2]

    for t in range(500):
        pdf = iteratePDF(pdf)
        print(t)
        if t % 2 == 0 or t < 100:
            continue

        pdf_plot = pdf[::2]
    
        fig, ax = plt.subplots(figsize=(10, 8), dpi=96)
        ax.plot(xvals, pdf_plot, c='k')
        ax.set_yscale("log")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r'$p_{\mathbf{B}}(x, t)$')
        ax.set_ylim([10**-20, 1])
        ax.set_xlim([-250, 250])
        #ax.set_yticks([10**-200, 10**-150, 10**-100, 10**-50, 10**0])
        ax.set_title(fr"$t={t}$")
        fig.savefig(f"./RWREPDF{t}.pdf", bbox_inches='tight')
        plt.close(fig)
