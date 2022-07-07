import numpy as np
import npquad 
from matplotlib import pyplot as plt
import glob
import os
import sys
sys.path.append("../../src")
import theory
from overalldatabase import Database
from TracyWidom import TracyWidom
from scipy.special import polygamma

varTW = 0.813

def x(theta, beta):
    return (polygamma(1, theta+2*beta) + polygamma(1, theta) - 2 * polygamma(1, theta+beta)) / (polygamma(1, theta) - polygamma(1, theta + 2 * beta))

def calculateMeanVar(files):
    mean = np.zeros(shape=(3299, 11))
    squared = np.zeros(shape=(3299, 11))
    count = 0
    for f in files: 
        print(f)
        data = np.loadtxt(f, skiprows=1, delimiter=',')
        if data[-1, 0] != 22701:
            continue
        mean+=data
        squared += data**2

        count+=1
    mean = mean / count
    var = squared / count - mean**2
    time = mean[:, 0] + 1
    return time, mean, var

def varPowerLaw(beta, t, N):
    logN = np.log(N).astype(float)
    return varTW * (t*logN/2)**(1/3) * beta**(-4/3)

dir = "/home/jacob/Desktop/talapasMount/JacobData/BetaCDFLongTime/"
dirs = os.listdir(dir)
run_again = False

'''Make plot of env'''
figv, axv = plt.subplots()
axv.set_xscale("log")
axv.set_yscale("log")
axv.set_xlabel("t / log(N)")
axv.set_ylabel("Var(Env)")
logN = np.log(1e24)
N = 1e24
colors = ['r', 'b', 'c', 'm']

for i, beta in enumerate([0.01, 0.1]):
    if run_again: 
        path = os.path.join(dir, str(beta), 'Q*.txt')
        files = glob.glob(path)
        time, mean, var = calculateMeanVar(files)
        np.savetxt(f"Data/Mean{beta}.txt", mean)
        np.savetxt(f"Data/Var{beta}.txt", var)
        np.savetxt(f"Data/Time{beta}.txt", time)
    mean_file = f"Data/Mean{beta}.txt"
    var_file = f"Data/Var{beta}.txt"
    time_file = f"Data/Time{beta}.txt"

    mean = np.loadtxt(mean_file)
    var = np.loadtxt(var_file)
    time = np.loadtxt(time_file)

    axv.plot(time / logN, var[:,3], label=beta, c=colors[i], alpha=0.75)
#axv.plot(time / logN, varPowerLaw(beta, time, N), c=colors[i], ls='--')
#axv.plot(time / logN, theory.quantileVarLongTimeBetaDist(N, time, beta), ls='-.', c=colors[i])

mathematica_time = np.loadtxt("Data/times.txt")
mathematica_varinace = np.loadtxt("Data/Variance.txt")
mathematica_varinace001 = np.loadtxt("Data/Variance001.txt")
theta_vals = np.loadtxt("Data/thetaVals.txt")

def getXvar(theta_vals, nSamples, beta):
    theta0 = theta_vals[:, 0]
    dtheta = theta_vals[:, 1]

    nSamples = 1000
    tw = TracyWidom(beta=2)
    r = np.random.rand(len(dtheta), nSamples)
    tw_sample = tw.cdfinv(r).T
    theta0 = np.tile(theta0, (nSamples, 1))
    dtheta = np.tile(dtheta, (nSamples, 1))
    theta = theta0 + tw_sample * dtheta
    xvals = x(theta, beta=beta)
    xvar = mathematica_time**2*np.var(xvals, axis=0)
    return xvar

xvar01 = getXvar(theta_vals, 1000, 0.1)
axv.plot(mathematica_time / logN, mathematica_varinace, c=colors[1], ls='--')
axv.plot(mathematica_time / logN, mathematica_varinace001, c=colors[0], ls='--')
#axv.plot(mathematica_time / logN, xvar01, ls='-.')

axv.set_xlim([min(time/logN), max(time/logN)])
axv.set_ylim([10**-2, 10**5])
axv.legend()
figv.savefig("Var.png")