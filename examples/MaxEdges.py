import sys

sys.path.append("../src")
from pydiffusionPDF as DiffusionPDF
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

# Specify some constants like the number of particles, beta, and number of
# timesteps to evolve the system
nParticles = 1e50
beta = 1
num_of_timesteps = 10_000

# Initialize the system with parameters and other key word arguments
d = DiffusionPDF(
    nParticles,
    beta=beta,
    occupancySize=num_of_timesteps,
    probDistFlag=False,
)

# Evolve the system to the specified number of timesteps
d.evolveToTime(num_of_timesteps)

# Get the rightmost edge and the time
maxEdge = d.maxDistance
time = d.time

# Plot the edge over time and save
fig, ax = plt.subplots()
ax.set_xlabel("Time")
ax.set_ylabel("Distance to Center")
ax.set_xscale("log")
ax.set_yscale("log")
ax.plot(time, maxEdge)
fig.savefig("MaxEdge.png")
