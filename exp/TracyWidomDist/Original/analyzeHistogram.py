import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import glob
import sys
sys.path.append("../../../src")
sys.path.append("../../../cDiffusion")
from pydiffusion import loadArrayQuad
import os
from TracyWidom import TracyWidom

file_dir = "/home/jhass2/Data/1.0/TracyWidomQuad2/"

files = glob.glob(file_dir + "T*.txt")
print('Number of files found:', len(files))
with open(files[0]) as g:
    vs = g.readline().split(",")[1:]
    vs = [float(i) for i in vs]

data = np.loadtxt(files[0], delimiter=",", skiprows=1)
shape = data.shape


squared_sum = None
reg_sum = None

run_again = False

# rows are different systems and colums are different vs
total_data = np.empty((len(files), shape[1] - 1))

if run_again:
    count = 0
    for row, f in enumerate(files):
        try:
            data = loadArrayQuad(f, (1, shape[1]), skiprows=int(shape[0]), delimiter=",")
        except Exception as e:
            print('File went wrong: ', f)
            print(e)
            continue

        time = data[-1, 0].astype(np.float64)
        data = data[-1, 1:]
        data = np.log(data).astype(np.float64)
        total_data[row] = data
        print(f)

    np.savetxt(file_dir + 'FinalTime.txt', total_data)

else:
    total_data = np.loadtxt(file_dir + 'FinalTime.txt')
    data = loadArrayQuad(files[0], (1, shape[1]), skiprows=int(shape[0]), delimiter=",")
    time = data[-1, 0].astype(np.float64)

print('At t: ', time)
tw = TracyWidom(beta=2)
for i in range(len(vs)):
    v = vs[i]
    data = total_data[:, i]
    data = data[~np.isinf(data)]
    data = data[~np.isnan(data)]
    fig, ax = plt.subplots()
    I = 1 - np.sqrt(1 - v**2)
    sigma = ((2 * I**2) / (1-I)) ** (1/3)
    scale = time ** (1/3) * sigma
    offset = I * time
    ax.set_xlabel("(ln(Pb(vt, t)) + I(v)t) / t^(1/3) * sigma")
    ax.set_ylabel("Probability Density")
    ax.set_title(f"v={v}")
    ax.hist((data + offset)/scale, density=True, bins=100)
    x = np.linspace(-5, 5, 100)
    pdf = tw.pdf(x)
    ax.plot(x, pdf, label='Theory')
    ax.set_yscale('log')
    fig.savefig(f"./histograms/hist{v}.png", bbox_inches='tight')
    plt.close(fig)

vs = np.array(vs)
lnPb = np.mean(total_data, axis=0)
I = 1 - np.sqrt(1-vs**2) 
sigma = ((2*I**2) / (1-I))**(1/3)
scale = time ** (1/3) * sigma
offset = I * time
fig, ax = plt.subplots()
ax.set_xlabel("v")
ax.set_ylabel("(ln(Pb(vt, t) + I(v)t) / t^(1/3) * sigma")
ax.set_xscale("log")
ax.set_yscale("log")
data = (lnPb + offset) / scale
ax.scatter(vs, abs(data))
ax.hlines(1.77, min(vs), max(vs))
fig.savefig("./histograms/means.png", bbox_inches="tight")

