import numpy as np
from matplotlib import pyplot as plt
import matplotlib.collections as mc
import pandas as pd

L = 50
r = 0.2
num = 5000
sizes = r * np.ones(num)
xy = np.random.uniform(-L /2, L/2, (num, 2))

df = pd.DataFrame(np.array([xy[:, 0], xy[:, 1], sizes]).T, columns=['x', 'y', 'r'])
df['Distance'] = np.sqrt(df['x']**2 + df['y']**2)
df = df[df['Distance'] >= df['r']]
df.drop(columns='Distance', inplace=True)
df.reset_index(inplace=True)

xy = np.array([df['x'].values, df['y'].values]).T

def xyOverlaps(xy, df):
    distance = np.sqrt((df['x'].values-xy[0])**2 + (df['y'].values - xy[1])**2)
    return np.any(distance <= df['r'].values)

def iterateTimeStep(xy, df, step_size=0.05):
    dx = np.random.normal(loc=(0, 0), scale=(step_size, step_size))
    xy_new = xy + dx
    if xyOverlaps(xy_new, df):
        return iterateTimeStep(xy, df, step_size)
    else:
        return xy_new

timesteps=10000
particle_path = np.array([0, 0]).reshape(1, 2)

for _ in range(timesteps):
    xy_new = iterateTimeStep(particle_path[-1, :], df, step_size=0.1)
    particle_path = np.vstack([particle_path, xy_new])
    #print(particle_path)

# Note that the patches won't be added to the axes, instead a collection will
patches = [plt.Circle(center, size) for center, size in zip(xy, sizes)]

fig, ax = plt.subplots()

coll = mc.PatchCollection(patches, facecolors='black')
ax.add_collection(coll)
#ax.scatter(particle_path[:, 0], particle_path[:, 1], c=np.arange(len(particle_path[:, 0])), s=0.5, linewidths=0)
ax.plot(particle_path[:, 0], particle_path[:, 1], c='r', lw=0.5)
ax.margins(0.01)
ax.set_xlim([-L/2, L/2])
ax.set_ylim([-L/2, L/2])
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_title(fr"$N={num}$")
fig.savefig("Gas.pdf", bbox_inches='tight')