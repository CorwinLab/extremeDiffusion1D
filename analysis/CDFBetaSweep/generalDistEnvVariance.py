import numpy as np 
import npquad
import os
import glob
import sys 
import matplotlib 
matplotlib.rcParams.update({'font.size': 14})

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.special import digamma

sys.path.append("../../dataAnalysis")
from overalldatabase import Database
from theory import loglog_moving_average, quantileVarLongTimeBetaDist

delta_color = 'tab:blue'
beta_color = 'tab:red'
inv_triang_color = 'tab:purple'
quad_color = 'tab:orange'
bates_color = 'tab:pink'
uniform_color = 'tab:cyan'

Nexp = 24

def plot_dirs(dirs, colors, labels, Nexp, beta, var, c2, save_file):
    N = float(f"1e{Nexp}")
    logN = np.log(N)
    
    alpha = 0.75

    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$t / \log(N)$")
    ax.set_ylabel(r"$\mathrm{Var}(\mathrm{Env}^N_t)$")
    ax.set_title(fr"$\sigma^2_w = {var}, c_2 = {c2}$")

    for i, d in enumerate(dirs):
        db = Database()
        db.add_directory(d, dir_type='Gumbel')
        cdf_df, _ = db.getMeanVarN(Nexp)
        ax.plot(cdf_df['time'] / logN, cdf_df['Var Quantile'], alpha=alpha, c=colors[i], label=labels[i])

    theory_var = quantileVarLongTimeBetaDist(N, cdf_df['time'].values, beta)
    ax.plot(cdf_df['time'] / logN, theory_var, ls='--', c='k')

    ax.set_xlim([1/np.log(N), 5000])
    
    leg = ax.legend(
        framealpha=0,
        labelcolor=colors,
        handlelength=0,
        handletextpad=0,
    )
    for item in leg.legendHandles:
        item.set_visible(False)
    fig.savefig(save_file, bbox_inches='tight')


beta100_dirs = ["/home/jacob/Desktop/corwinLabMount/CleanData/CDFBetaSweep/100/",
                "/home/jacob/Desktop/talapasMount/JacobData/Bates/804",
                "/home/jacob/Desktop/talapasMount/JacobData/Uniform/804",
                "/home/jacob/Desktop/talapasMount/JacobData/Delta/804",
                "/home/jacob/Desktop/talapasMount/JacobData/Quadratic/804"]
beta100_colors = [beta_color, bates_color, uniform_color, delta_color, quad_color]
labels = ['Beta', 'Bates', 'Uniform', 'Delta', 'Quadratic']
beta100_save_file = "Beta100Quantile.pdf"
beta=100
var = r"\frac{1}{804}"
c2 = r"10^4"

plot_dirs(beta100_dirs, beta100_colors, labels, Nexp, beta, var, c2, beta100_save_file)

beta10_dirs = ["/home/jacob/Desktop/corwinLabMount/CleanData/CDFBetaSweep/10/",
                "/home/jacob/Desktop/talapasMount/JacobData/Bates/84",
                "/home/jacob/Desktop/talapasMount/JacobData/Uniform/84",
                "/home/jacob/Desktop/talapasMount/JacobData/Delta/84",
                "/home/jacob/Desktop/talapasMount/JacobData/Quadratic/84"]
beta10_colors = [beta_color, bates_color, uniform_color, delta_color, quad_color]
labels = ['Beta', 'Bates', 'Uniform', 'Delta', 'Quadratic']
beta10_save_file = "Beta10Quantile.pdf"
beta=10
c2=r"10^2"

plot_dirs(beta10_dirs, beta10_colors, labels, Nexp, beta, var, c2, beta10_save_file)

beta1_dirs = ["/home/jacob/Desktop/corwinLabMount/CleanData/CDFBetaSweep/1/",
                "/home/jacob/Desktop/talapasMount/JacobData/Delta/12",
                "/home/jacob/Desktop/talapasMount/JacobData/Quadratic/12"]
labels = ['Beta', 'Delta', 'Quadratic']
beta1_colors = [beta_color, delta_color, quad_color]
beta1_save_file = "Beta1Quantile.pdf"
beta=1
c2=r"1"

plot_dirs(beta1_dirs, beta1_colors, labels, Nexp, beta, var, c2, beta1_save_file)

beta01_dirs = ["/home/jacob/Desktop/corwinLabMount/CleanData/CDFBetaSweep/0.1/",
                "/home/jacob/Desktop/talapasMount/JacobData/Delta/512",
                "/home/jacob/Desktop/talapasMount/JacobData/InvTriangle/524"]
beta01_colors = [beta_color, delta_color, inv_triang_color]
labels = ['Beta', 'Delta', 'Comp. Triangular']
beta01_save_file = "Beta01Quantile.pdf"
beta=0.1
c2=r"10^{-2}"

plot_dirs(beta01_dirs, beta01_colors, labels, Nexp, beta, var, c2, beta01_save_file)

beta01_dirs = ["/home/jacob/Desktop/corwinLabMount/CleanData/CDFBetaSweep/0.01/",
                "/home/jacob/Desktop/talapasMount/JacobData/Delta/25102",
                "/home/jacob/Desktop/talapasMount/JacobData/InvTriangle/25102"]
beta01_colors = [beta_color, delta_color, inv_triang_color]
labels = ['Beta', 'Delta', 'Comp. Triangular']
beta01_save_file = "Beta001Quantile.pdf"
beta=0.01
c2=r"10^{-4}"

plot_dirs(beta01_dirs, beta01_colors, labels, Nexp, beta, var, c2, beta01_save_file)