from continuousDiffusion import * 
import matplotlib.pyplot as plt

sampled = sampleBiasFromBiasingField(arange(-10, 10, 0.5), 2, 0.1)

fig, ax = plt.subplots()
ax.plot(arange(-10, 10, 0.5), sampled, marker='*')
fig.savefig("SampledFromField.png")