import continuousDiffusion
import matplotlib.pyplot as plt

system = continuousDiffusion.System(10000)
tMax = 10000
for _ in range(tMax):
    system.iterateTimeStep()
positions=system.getParticlePositions()
print(max(positions))
fig, ax = plt.subplots()
ax.hist(positions, bins=50)
fig.savefig("Positions.png", bbox_inches='tight')

print(range)