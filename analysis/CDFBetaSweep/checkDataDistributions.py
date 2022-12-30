import numpy as np
import os 
import json
import sys 
sys.path.append("../../src")
from pyDiffusion import DiffusionTimeCDF

delta_dir = "/home/jacob/Desktop/talapasMount/JacobData/Delta/"
dirs = ['12', '84', '804']
file = "RandomNums0.txt"
print("Delta Distribution:")
for d in dirs: 
    f = os.path.join(delta_dir, d, file)
    data = np.loadtxt(f)
    print("----------------")
    print(f"Delta:", np.var(data))
    print("Real Value:", 1/float(d))
    print("Percent Difference:", np.abs((np.var(data) - 1/float(d)) / (1/float(d))*100), "%")

print("\n")
print("Bates Distribution:")
bates_dir = "/home/jacob/Desktop/talapasMount/JacobData/Bates/"
dirs = ['84']
file = "RandomNums0.txt"

for d in dirs:
    f = os.path.join(bates_dir, d, file)
    data = np.loadtxt(f)
    print("----------------")
    print(f"Bates:", np.var(data))
    print("Real Value:", 1/float(d))
    print("Percent Difference:", np.abs((np.var(data) - 1/float(d)) / (1/float(d))*100), "%")

print("----------------")
d = '804'
f = open(os.path.join(bates_dir, d, 'variables.json'))
vars = json.load(f)
a = vars['a']
b = vars['b']
n = vars['n']
f.close()

rec = DiffusionTimeCDF('bates', [n, a, b], 1000)
data = []
for _ in range(100000):
    data.append(rec.generateRandomVariable())

print(f"Bates:", np.var(data))
print("Real Value:", 1/float(d))
print("Percent Difference:", np.abs((np.var(data) - 1/float(d)) / (1/float(d))*100), "%")
