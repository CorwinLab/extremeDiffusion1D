import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
	width = 1000
	mu = np.ones(2 * width + 1)
	
	num_steps = 1000
	# Edge Case
	# pii = 6/16 
	# pjj1 = 1/4
	# pjj2 = 1/16
	# p01 = 11/48

	# Dirichlet with alpha = 1
	# pii = 1 / 3 
	# pjj1 = 2 / 9
	# pjj2 = 1 / 9
	# p01 = 1 / 6
	# p02 = 1 / 12
	# c = 3 / 4 

	# Symmetric with uniform(0, 1/2)
	pii = 1/4 
	pjj1 = 1/4
	pjj2 = (1 - pii - 2*pjj1) / 2
	p01 = 1 / 6
	p02 = 1 / 12
	
	print(1 - pii, 2 * pjj1 + 2 * pjj2)

	for _ in range(num_steps):
		mu_new = np.zeros(len(mu))
		for i in range(2, len(mu) - 2):
			if i == (len(mu) // 2):
				mu_new[i] = 1
				continue
			elif i == ((len(mu) // 2) - 1):
				mu_new[i] = pii * mu[i] + pjj1 * (mu[i-1]) + pjj2 * (mu[i+2] + mu[i-2]) + mu[i+1] * p01
				continue
			elif i == ((len(mu) // 2) + 1):
				mu_new[i] = pii * mu[i] + pjj1 * (mu[i+1]) + pjj2 * (mu[i+2] + mu[i-2]) + mu[i-1] * p01
				continue
			elif i == ((len(mu) //2 ) + 2):
				mu_new[i] = pii * mu[i] + pjj1 * (mu[i+1] + mu[i-1]) + pjj2 * (mu[i+2]) + mu[i-2] * p02
				continue
			elif i == ((len(mu) // 2) - 2):
				mu_new[i] = pii * mu[i] + pjj1 * (mu[i+1] + mu[i-1]) + pjj2 * (mu[i-2]) + mu[i+2] * p02
				continue
			mu_new[i] = pii * mu[i] + pjj1 * (mu[i+1] + mu[i-1]) + pjj2 * (mu[i+2] + mu[i-2])
		mu = mu_new
		
def expected_func(xvals):
	'''Expected function for edge case example'''
	c1 = (12 * np.sqrt(2) -1) / (12 * np.sqrt(2))
	c3 = 1 - c1 
	return c1 + c3 * (2 * np.sqrt(2) - 3) ** np.abs(xvals)

fig, ax = plt.subplots()
ax.set_xlim([-200, 200])
xvals = np.arange(-width, width + 1)
ax.scatter(xvals, mu)
ax.set_title("Symmetric Distribution")
ax.set_xlabel("l")
ax.set_ylabel(r"$\mu(l)$")
# ax.set_yscale("log")
# ax.set_ylim([10**-44, 1])
ax.set_xlim([-10, 10])
# ax.plot(xvals, expected_func(xvals), ls='--', c='r')
# ax.hlines(c, -width, width, ls='--', color='r')
# ax.plot(xvals, np.exp(-np.abs(xvals) / 1.75), c='k')
fig.savefig("mu.png", bbox_inches='tight')