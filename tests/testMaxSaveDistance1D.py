from pyDiffusion.pydiffusion2D import evolveAndSaveMaxDistance1D
import numpy as np

nParticles = 10_000
save_times = np.linspace(0, 10_000, 50).astype(int)
save_times = np.unique(save_times)
xi = 1 
d=0.1
sigma = 1
save_file = 'MaxTest.txt'
save_positions = 'TestPositions.txt'

evolveAndSaveMaxDistance1D(nParticles, save_times, xi, d, sigma, save_file, save_positions)