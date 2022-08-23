import numpy as np
from matplotlib import pyplot as plt
from numba import jit

@jit(nopython=True)
def gaussian(A, mean, sigma, x):
    return (
        A / np.sqrt(2 * np.pi * sigma**2) * np.exp(-((x - mean) / sigma) ** 2 / 2)
    )

@jit(nopython=True)
def iterateTimeStep(particle_coords, correlation_length):
    # Should pass in density instead and then
    # number_of_points = int(density * L**dim)
    dim = particle_coords.shape[1] 
    
    # generate random field
    L = np.max(particle_coords)
    number_of_points = np.random.poisson(int(L ** dim))
    gaussian_center_coords = np.random.uniform(-L, L, size=(number_of_points, dim))
    gaussian_amplitudes = np.random.normal(0, 0.5, size=(number_of_points))
    # Not needed except for debugging
    biases = np.zeros(shape=particle_coords.shape)
    
    # Iterate all of the particles by a single step in time
    for i in range(particle_coords.shape[0]):
        # Get the bias for each particle
        bias = np.zeros(shape=dim)
        # This should either be vectorized or numba-ized
        # bias = gaussian(gaussian_amplitudes, gaussian_center_coords, correlation_length, particle_coords)
        for j in range(number_of_points):
            bias += gaussian(
                gaussian_amplitudes[j], gaussian_center_coords[j, :], correlation_length, particle_coords[i, :]
            )
        # Draw step from random gaussian with mean=bias and sigma=correlation_length
        displacement = np.array([np.random.uniform(i, correlation_length) for i in bias])
        biases[i, :] = bias
        particle_coords[i, :] += displacement
    
    return particle_coords, biases


if __name__ == "__main__":
    # Generate the random field
    d = 2
    correlation_length = 1

    # Start all particles at x=0
    number_of_particles = 1000
    xx, yy = np.meshgrid(np.arange(-10, 10, 1), np.arange(-10, 10, 1))
    particle_coords = np.array([xx.flatten(), yy.flatten()]).T.astype(float)

    particle_coords2, biases = iterateTimeStep(particle_coords.copy(), correlation_length)
    
    fig, ax = plt.subplots()
    ax.scatter(particle_coords[:, 0], particle_coords[:, 1], s=4)
    ax.quiver(particle_coords[:, 0], particle_coords[:, 1], biases[:, 0], biases[:, 1])
    fig.savefig("Field.pdf", bbox_inches='tight')

    particle_coords = np.zeros(shape=(1000, d))
    for i in range(100):
        particle_coords, biases = iterateTimeStep(particle_coords, correlation_length)
        print(i)
    
    fig, ax = plt.subplots()
    ax.scatter(particle_coords[:, 0], particle_coords[:, 1], s=4)
    fig.savefig("ParticleLocation.pdf", bbox_inches='tight')