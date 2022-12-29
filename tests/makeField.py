import numpy as np
from matplotlib import pyplot as plt

def getGCF1D(positions, correlation_length, D, grid_spacing=0.1):
    '''
    Generate a gaussian correlated field at given positions and correlation length.  

    Paramters
    ---------
    positions : numpy array (1D)
        Positions of particles
    
    correlation_length : float 
        Correlation length in space of the field 
    
    D : float 
        Diffusion coefficient to use 
    
    grid_spacing : float (optional 0.1)
        Grid spacing of the random field to generate. 
    
    Returns
    -------
    field : numpy array
        Field stregnth at each particle position.
    
    Example
    -------
    from matplotlib import pyplot as plt
    x = np.random.normal(0, 100, size=100000)
    correlation_length = 10
    field = getGCF1D(x, correlation_length=correlation_length, D=1, grid_spacing=0.1)
    fig, ax = plt.subplots()
    ax.scatter(x, field)
    ax.set_xlabel("Position")
    ax.set_ylabel("Field Strength")
    ax.set_title(f"Correlation Length = {correlation_length}")
    fig.savefig(f"TestingField{correlation_length}.png", bbox_inches='tight')
    '''

    noise_points = (np.max(positions) - np.min(positions) + 6 * correlation_length) / grid_spacing
    grid = np.linspace(np.min(positions) - 3 * correlation_length, np.max(positions) + 3 * correlation_length, int(noise_points))
    noise = np.random.randn(int(noise_points))
    
    kernel_x = np.arange(-3 * correlation_length, 3 * correlation_length + 0.1, grid_spacing)
    kernel = np.sqrt(D/correlation_length / np.sqrt(np.pi)) * np.exp(-kernel_x**2/2/correlation_length**2)
    noise = np.convolve(noise, kernel, 'same')

    field = np.interp(positions, grid, noise)
    return field