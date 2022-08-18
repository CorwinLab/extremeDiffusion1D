import numpy as np
from matplotlib import pyplot as plt

def gaussian(A, mean, sigma, x):
    return (
        A / np.sqrt(2 * np.pi * sigma**2) * np.exp(-((x - mean) / sigma) ** 2) / 2
    )


# Generate the random field
L = 100
r = 50
d = 2
number_of_points = r * L
random_coords = np.random.uniform(-L / 2, L / 2, size=(number_of_points, d))
random_amplitude = np.random.uniform(-1, 1, size=(number_of_points))
correlation_length = 2

# Start all particles at x=0
number_of_particles = 1000
particle_coords = np.zeros(shape=(number_of_particles, d))

biases = np.zeros(shape = particle_coords.shape)
# Iterate the particles in time
for i in range(len(particle_coords[:, 0])):
    # Get the bias for each particle
    bias = np.zeros(shape=d)
    for j in range(len(random_coords)):
        bias += gaussian(
            random_amplitude[j], random_coords[j, :], correlation_length, particle_coords[i, :]
        )
    # Draw step from random gaussian with mean=bias and sigma=correlation_length
    dx = np.random.normal(bias, scale=correlation_length)
    biases[i, :] = bias
    particle_coords[i, :] += dx
    print(i, 1)

plt.figure()
plt.hist(particle_coords[:, 0])
plt.savefig("test1.png")

random_coords = np.random.uniform(-L / 2, L / 2, size=(number_of_points, d))
random_amplitude = np.random.uniform(-1, 1, size=(number_of_points))
biases = np.zeros(shape=particle_coords.shape)
# Iterate the particles in time
for i in range(len(particle_coords[:, 0])):
    # Get the bias for each particle
    bias = np.zeros(shape=d)
    for j in range(len(random_coords)):
        bias += gaussian(
            random_amplitude[j], random_coords[j, :], correlation_length, particle_coords[i, :]
        )
    # Draw step from random gaussian with mean=bias and sigma=correlation_length
    dx = np.random.normal(bias, scale=correlation_length)
    biases[i, :] = bias
    particle_coords[i, :] += dx
    print(i, 2)

plt.figure()
plt.scatter(particle_coords[:, 0], particle_coords[:, 1])
plt.savefig("test2.png")

plt.figure()
plt.scatter(biases[:, 0], biases[:, 1])
plt.savefig("biases.png")