import numpy as np
from matplotlib import pyplot as plt

k = np.linspace(0, 100, 500)
correlation_length = 2
fourier_space = np.exp(-k**2 * correlation_length**2 / 4) * np.random.normal(0, 1/2/np.pi, len(k)) * np.exp(1j * np.random.uniform(0, 2*np.pi, len(k)))
field = np.fft.ifft(fourier_space)
print(field)
fig, ax = plt.subplots()
ax.scatter(range(0, len(field)), field.real)
fig.savefig("TestField.png")