import sys 
sys.path.append("./src")
from libDiffusion import ParticleData
import numpy as np 
import npquad

p1 = ParticleData(1000)
p2 = ParticleData(1001)
print(p1==p2)

p3 = ParticleData(1000)
print(p1==p3)