import numpy as np
from matplotlib import pyplot as plt
import matplotlib.collections as mc
import pandas as pd
from numba import njit

L = 500
r = 1
num = 10000
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

def iterateTimeStep(xy, v, df, step_size=0.01, dt=0.01):
    # Move all the particles in the gas
    dx = np.random.normal(loc=(0, 0), scale=(1, 1), size=(df['x'].size, 2))
    scaling = np.sqrt(dx[:, 0]**2 + dx[:, 1]**2)
    dx[:, 0] /= (scaling / step_size)
    dx[:, 1] /= (scaling / step_size)
    df['x'] += dx[:, 0] 
    df['y'] += dx[:, 1]

    # Now iterate the particles in the gas
    # Note that xy is an array with rows as particle positions
    # The velocities should be the same shape as well
    xy_new = xy + v * dt
    for i in range(0, xy_new.shape[0]):
        if xyOverlaps(xy_new[i, :], df):
            v[i, :] = np.random.randn(2)
            v[i, :] /= np.sqrt(np.sum(v[i, :]**2))
            
            xy_new[i, :] = xy[i, :] + v[i, :] * step_size
    
    return xy_new, v, df

timesteps=10000
particle_pos = np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]).astype(float)
v = np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]).astype(float)

# Keep track the paths of each particle here
# Let the indeces be (x, y, id)
particle_path_0 = particle_pos[0, :]
particle_path_1 = particle_pos[1, :]
particle_path_2 = particle_pos[2, :]
particle_path_3 = particle_pos[3, :]
particle_path_4 = particle_pos[4, :]

for t in range(timesteps):
    particle_pos, v, df = iterateTimeStep(particle_pos, v, df, step_size=0.5, dt=1)
    
    particle_path_0 = np.vstack([particle_path_0, particle_pos[0, :]])
    particle_path_1 = np.vstack([particle_path_1, particle_pos[1, :]])
    particle_path_2 = np.vstack([particle_path_2, particle_pos[2, :]])
    particle_path_3 = np.vstack([particle_path_3, particle_pos[3, :]])
    particle_path_4 = np.vstack([particle_path_4, particle_pos[4, :]])

    if t % 10 == 0:
        # Note that the patches won't be added to the axes, instead a collection will
        patches = [plt.Circle((x, y), size) for x, y, size in zip(df['x'].values, df['y'].values, sizes)]
        dpi = 96
        fig, ax = plt.subplots(figsize=(800/dpi, 800/dpi), dpi=dpi)

        coll = mc.PatchCollection(patches, facecolors='black')
        ax.add_collection(coll)
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
        ax.plot(particle_path_0[:, 0], particle_path_0[:, 1], c=colors[0], lw=2)
        ax.plot(particle_path_1[:, 0], particle_path_1[:, 1], c=colors[1], lw=2)
        ax.plot(particle_path_2[:, 0], particle_path_2[:, 1], c=colors[2], lw=2)
        ax.plot(particle_path_3[:, 0], particle_path_3[:, 1], c=colors[3], lw=2)
        ax.plot(particle_path_4[:, 0], particle_path_4[:, 1], c=colors[4], lw=2)

        ax.set_xlim([-L/2, L/2])
        ax.set_ylim([-L/2, L/2])
        ax.axis("off")
        fig.savefig(f"./MultipleParticlePlots/Gas{t}.pdf")
        plt.close(fig)
        print(t)