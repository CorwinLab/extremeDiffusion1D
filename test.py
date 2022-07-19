import numpy as np 
from matplotlib import pyplot as plt
from scipy.special import erf

def pdf(t, L):
    sum = 0
    for k in range(-1000, 1000):
        sum += 2 * np.sqrt(2/t**3 / np.pi) * L * (-1)**(np.abs(k)) * (k+1/2) * np.exp(-2* L**2 * (k+1/2)**2/t)
    return sum
    
def myfunction(t, L): 
    sum = 0
    for k in range(-1000, 1000):
        sum += - (-1)**abs(k) * (k+1/2) / np.abs(k+1/2) * erf(np.sqrt(2 * L**2 * (k+1/2)**2 / t))
    return sum

t = np.arange(1, 10000)
L = 50
fig, ax = plt.subplots()
ax.plot(t, myfunction(t, L))
fig.savefig("Test.png")

print(sum(pdf(t, L)))
fig, ax = plt.subplots()
ax.plot(t, pdf(t, L))
fig.savefig("PDF.png")