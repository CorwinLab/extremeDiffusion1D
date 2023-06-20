import numpy as np 
from matplotlib import pyplot as plt

def forwardEuler(f, dx, dt, D):
	f_new = np.zeros(f.shape)
	for row in range(1, f.shape[0]-1):
		for col in range(1, f.shape[1]-1):
			gradient = 1 / 6 / dx**2 * (4 * (f[row-1][col] + f[row+1][col] + f[row][col-1] + f[row][col+1]) + (f[row-1][col-1] + f[row-1][col+1] + f[row+1][col-1] + f[row+1][col+1]) - 20*f[row][col])
			f_new[row][col] += D * gradient * dt + f[row][col]
	return f_new 

def gaussian(xvals, yvals, var):
    return np.sqrt(1/(2 * np.pi) / np.sqrt(var)) * np.exp(-(xvals**2 + yvals**2)/2/var)

if __name__ == '__main__': 
	dx = 0.1
	xvals = np.arange(-5, 5+dx, dx)
	yvals = np.arange(-5, 5+dx, dx)
	D = 1
	t = 1
	dt = dx**2 / 4 / D
    
	xx, yy = np.meshgrid(xvals, yvals)
	gaus = gaussian(xx, yy, 2 * D * t)
    
	fig, ax = plt.subplots()
	cax = ax.contourf(xx, yy, gaus)
	fig.colorbar(cax)
	fig.savefig("Gauss.png", bbox_inches='tight')
    
	f = gaussian(xx, yy, 2 * D * t)
	
	for t in range(1000):
		f = forwardEuler(f, dx, dt, D)

	fig, ax = plt.subplots()
	cax = ax.contourf(xx, yy, f)
	fig.colorbar(cax)
	fig.savefig("Iterated.png", bbox_inches='tight')