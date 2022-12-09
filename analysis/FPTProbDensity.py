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

def iteratePDF(pdf):
    biases = np.random.uniform(0, 1, size=pdf.shape)
    pdf_new = np.zeros(pdf.shape)
    for i in range(1, len(pdf)-1):
        pdf_new[i] = pdf[i-1] * biases[i-1] + pdf[i+1] * (1- biases[i+1])
    return pdf_new

if __name__ == '__main__':
    maxPosition = 500
    pdf = initializePDF(maxPosition)
    for _ in range(500):
        pdf = iteratePDF(pdf)
    print(sum(pdf))
    xvals = np.arange(-maxPosition, maxPosition+1, 1)
    fig, ax = plt.subplots()
    xvals = xvals[::2]
    pdf = pdf[::2]
    ax.plot(xvals, pdf, c='k')
    ax.set_yscale("log")
    ax.set_xlabel("x")
    ax.set_ylabel(r'$p^{\infty}_{\mathbf{B}}(x, t)$')
    ax.set_yticks([10**-200, 10**-150, 10**-100, 10**-50, 10**0])
    L = 200 
    lower_fill_x = xvals[xvals < -L]
    lower_fill_prob = pdf[xvals < -L]
    upper_fill_x = xvals[xvals > L]
    upper_fill_prob = pdf[xvals > L]
    indeces = (-L < xvals) * (xvals < L)
    new_xvals = xvals[indeces]
    new_pdf = pdf[indeces]
    new_xvals = np.insert(new_xvals, 0, -L)
    new_xvals = np.append(new_xvals, L)
    new_pdf = np.insert(new_pdf, 0, sum(lower_fill_prob))
    new_pdf = np.append(new_pdf, sum(upper_fill_prob))
    #ax.plot(new_xvals, new_pdf)
    ax.fill_between(lower_fill_x, lower_fill_prob, np.ones(lower_fill_prob.shape)*min(lower_fill_prob), alpha=0.5, color='r', label=r'$P^{\infty}_{\mathbf{B}}(X > L, t)$')
    ax.fill_between(upper_fill_x, upper_fill_prob, np.ones(upper_fill_prob.shape)*min(upper_fill_prob), alpha=0.5, color='r', label=r'$P^{\infty}_{\mathbf{B}}(X < -L, t)$')
    ax.legend()
    fig.savefig("RWREPDF.pdf", bbox_inches='tight')