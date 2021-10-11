from matplotlib import pyplot as plt
import numpy as np
import npquad
import glob

files = glob.glob("/home/jacob/Desktop/corwinLabMount/Data/Einstein/1.00e_50/Edges*.txt")
mean_sum = None
squared_sum = None
run_again = False
if run_again:
    for f in files:
        data = np.loadtxt(f)
        time = np.arange(1, len(data[:,0])+1)
        center = time * 0.5
        max_disp = 2 * (data[:, 1] - center)
        if mean_sum is None:
            mean_sum = np.zeros(len(center))
            squared_sum = np.zeros(len(center))
        mean_sum += max_disp
        squared_sum += max_disp ** 2
        print(f)

    var = squared_sum / len(files) - (mean_sum / len(files)) ** 2
    np.savetxt("Variance.txt", var)
    np.savetxt("Time.txt", time)

time = np.loadtxt("Time.txt")
var = np.loadtxt("Variance.txt")
logN = np.log2(1e5)

fig, ax = plt.subplots()
ax.set_xlabel("Time")
ax.set_ylabel("Variance")
ax.set_xscale("log")
ax.set_yscale("log")
ax.plot(time / logN, var)
fig.savefig("Variance.png")
