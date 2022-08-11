from pyDiffusion import FirstPassageDriver
import numpy as np 
import npquad

N = 1e24
distances = np.geomspace(10, 500*np.log(N), num=500).astype(int)
distances = np.unique(distances)
print(distances)
pdf = FirstPassageDriver(1, distances)
pdf.evolveToCutoff(N, 'test.csv', 1, True)