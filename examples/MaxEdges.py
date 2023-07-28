from pyDiffusion import DiffusionPDF
from matplotlib import pyplot as plt
import numpy as np

# Specify some constants like the number of particles, beta, and number of
# timesteps
nParticles = 1e50
distributionName = 'beta'
beta = 1
num_of_timesteps = 1_000

# Initialize the system with parameters and other key word arguments
d = DiffusionPDF(
    nParticles,
    distributionName=distributionName,
    parameters=[beta, beta],
    occupancySize=num_of_timesteps,
    ProbDistFlag=False,
)

maxEdge = np.zeros(num_of_timesteps)
for t in range(num_of_timesteps):
    d.iterateTimestep()
    maxEdge[t] = 2 * d.edges[1] - d.currentTime

# Plot the edge over time and save
fig, ax = plt.subplots()
ax.set_xlabel("Time")
ax.set_ylabel("Distance to Center")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim([1, num_of_timesteps])
ax.plot(d.time[1:], maxEdge)
fig.savefig("MaxEdge.png")
