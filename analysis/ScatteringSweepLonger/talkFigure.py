import numpy as np 
from matplotlib import pyplot as plt
import os 
import sys 
sys.path.append("../../dataAnalysis")
from overalldatabase import Database

b = 1
Nexp = 7
dir = f'/home/jacob/Desktop/talapasMount/JacobData/ScatteringNSweep/{Nexp}'
lw = 1
N = float(f"1e{Nexp}")
logN = np.log(N)

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$t / \log(N)$")
ax.set_ylabel(r"$\mathrm{Var}(\mathrm{Env}^N_t)$")
ax.set_xlim([0.3, 5 * 10**3])
ax.set_ylim([10**-1, 10**3])

# Plot the scattering model variance
var = np.loadtxt(os.path.join(dir, "Var.txt"))
time = np.loadtxt(os.path.join(dir, "Time.txt"))

ax.plot(time / logN, var, c='tab:red', label='Scattering Model')

# Plot RWRE model variance
db = Database()
cdf_path = "/home/jacob/Desktop/corwinLabMount/MaxParticlePaperData/Paper/CDF/"
cdf_path_talapas = "/home/jacob/Desktop/corwinLabMount/MaxParticlePaperData/JacobData/Paper/"

db.add_directory(cdf_path, dir_type="Gumbel")
db.add_directory(cdf_path_talapas, dir_type="Gumbel")

db1 = db.getBetas(1)
cdf_df, _ = db1.getMeanVarN(Nexp)

ax.plot(cdf_df['time'] / logN, cdf_df['Var Quantile'], c='tab:blue', label='RWRE')

xvals = np.array([3* 10 ** 2, 4 * 10**3])
ax.plot(xvals, xvals**(1/2), ls='--', c='k', label=r'$t^{1/2}$')

leg = ax.legend(
    loc="upper right",
    framealpha=0,
    labelcolor=['tab:red', 'tab:blue', 'k'],
    handlelength=0,
    handletextpad=0,
)
for item in leg.legendHandles:
    item.set_visible(False)

fig.savefig("ScatteringRWREComp.svg", bbox_inches='tight')