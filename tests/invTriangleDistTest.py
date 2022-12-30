import numpy as np
from matplotlib import pyplot as plt 

size=10_000
var = 1/8
c = 1 - np.sqrt(1-6*(1/4-var))
print(c)
rand_nums = np.random.uniform(0, 1, size=size)
rand_dist = np.zeros(size)

for i in range(len(rand_dist)):
    y = rand_nums[i]
    if y <= 1/2: 
        rand_dist[i] = c * (1 - np.sqrt(1-2*y))
    elif y > 1/2:
        rand_dist[i] = c * (np.sqrt(2 * (y-1/2))-1) + 1

print("Mean:", np.mean(rand_dist))
print("Var:", np.var(rand_dist))
print("Real Var:", var)
fig, ax = plt.subplots()
ax.hist(rand_dist, bins=50, density=True)
ax.set_xlim([0, 1])
fig.savefig("RandomNum.png", bbox_inches='tight')