import numpy as np
import npquad 
from matplotlib import pyplot as plt
import glob 
import os 

path = "/home/jacob/Desktop/talapasMount/JacobData/FirstPassDiscreteAbsRangeN"
dirs = os.listdir(path)
run_again = False

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylabel(r"$Var(\tau)$")
ax.set_xlabel(r"$\log(N)$")

if run_again: 
    Ns = []
    vars = []
    for d in dirs:
        N_exp = int(d)
        N = np.quad(f"1e{N_exp}")
        file_path = os.path.join(path, d, "Q*.txt")
        files = glob.glob(file_path)
        times = []
        for f in files:
            data = np.loadtxt(f, skiprows=1, delimiter=',')
            distance = data[0]
            times.append(data[1])
        Ns.append(N_exp)
        vars.append(np.var(times))
    np.savetxt("Var.txt", np.array([Ns, vars]))

else:
    data = np.loadtxt("Var.txt")
    Ns = data[0, :]
    vars = data[1, :]

N = np.array([np.quad(f"1e{i}") for i in Ns])
xvals = np.array([10**2, 10**3])
yvals = 1/xvals**4 * 10**12
ax.scatter(np.log(N).astype(float), vars, c='k', label='SSRW')
ax.plot(xvals, yvals, label=r'$(\log(N))^{-4}$')
ax.legend()
fig.savefig("VarianceOverN.pdf", bbox_inches='tight')