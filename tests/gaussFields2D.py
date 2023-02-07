import numpy as np 
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

'''
Okay this doesn't work. Picks out xy components
'''
x = np.linspace(0, 10)
y = np.linspace(0, 10)

xx, yy = np.meshgrid(x, y)
x = xx.flatten()
y = yy.flatten()

rand_vector = np.random.normal(0, scale=0.5, size=(xx.shape[0], xx.shape[1], 2))
fig, ax = plt.subplots()
ax.quiver(x, y, rand_vector[:, :, 0].flatten(), rand_vector[:, :, 1].flatten(), np.sqrt(rand_vector[:, :, 0].flatten()**2 + rand_vector[:, :, 1].flatten()**2), angles='xy', scale=1, scale_units='xy')
ax.grid(True)
fig.savefig("RandomField.png")

correlation_length = np.pi
x_field_comp = gaussian_filter(rand_vector[:, :, 0], correlation_length)
y_field_comp = gaussian_filter(rand_vector[:, :, 1], correlation_length)

fig, ax = plt.subplots()
ax.quiver(xx.flatten(), yy.flatten(), x_field_comp.flatten(), y_field_comp.flatten(), np.sqrt(x_field_comp.flatten()**2 + y_field_comp.flatten()**2))#, angles='xy', scale=1, scale_units='xy')
ax.grid(True)
fig.savefig("CorrelatedField.png")

def make_kernel(size_x, size_y, correlation_length): 
    x_range = np.arange((-size_x+1)/2, (size_x-1)/2+1, 1)
    y_range = np.arange((-size_y+1)/2, (size_y-1)/2+1, 1)
    xx, yy = np.meshgrid(x_range[::-1], y_range)
    distance = np.sqrt(xx**2 + yy**2)
    kernel = np.exp(-distance**2 / 2 / correlation_length ** 2 )
    return kernel

def generateGCDF2D(correlation_length):
    x = np.arange(0, 100)
    y = np.arange(0, 100)

    xx, yy = np.meshgrid(x, y)
    x = xx.flatten()
    y = yy.flatten()

    kernel = make_kernel(8 * correlation_length, 8 * correlation_length, correlation_length)

    rand_vector = np.random.normal(0, scale=0.5, size=(xx.shape[0], xx.shape[1], 2))
    x_field_comp = gaussian_filter(rand_vector[:, :, 0], correlation_length)
    y_field_comp = gaussian_filter(rand_vector[:, :, 1], correlation_length)

    return x_field_comp.flatten(), y_field_comp.flatten(), xx, yy

numSystems = 1
dot_product = None
for _ in range(numSystems): 
    x_comp, y_comp, xx, yy = generateGCDF2D(correlation_length)
    if dot_product is None: 
        dot_product = np.zeros(x_comp.shape)
    dot_product += x_comp[0] * x_comp[:] + y_comp[0] * y_comp[:]

dot_product /= numSystems
dot_product = dot_product.reshape(xx.shape)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(xx, yy, dot_product, linewidth=0, antialiased=True, cmap=plt.get_cmap('cool'), alpha=0.75)
ax.plot3D(xx[0, :], yy[0, :], dot_product[0, :], c='r')
fig.savefig("TwoPointCorrelator.png", bbox_inches='tight')

theoretical = max(dot_product[0, :]) * np.exp(-(xx[0, :] - xx[0, 0])**2 / 4 / correlation_length**2)

fig, ax = plt.subplots()
ax.plot(xx[0, :], dot_product[0, :], c='r')
ax.plot(xx[0, :], theoretical, c='k', ls='--')
fig.savefig("SliceCorrelator.png", bbox_inches='tight')
