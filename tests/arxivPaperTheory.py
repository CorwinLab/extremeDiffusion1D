import numpy as np
from scipy.special import zeta

gamma = np.euler_gamma

def moment_generating_function(N, C, D0, D1, D2, theta, m):
    beta = theta / C * np.log(C / theta * (D0*N)**(1/theta)) # approximation of w-lambert function
    bn = beta * (1 + D1 / C / beta**2 - (2 * theta * D1 + C*(D1**2 - 2*D2))/(2 * C**2 * beta**3) )
    an = 1/C * (1- theta/C/beta + (theta**2-C*D1) / C**2 / beta**2 - (2*theta**3 - 6 * theta * C * D1 - 2 * C**2 *(D1**2 - 2 * D2))/2/C**3 /beta**3)
    c2 = -theta / np.log(N)**2 
    c3 = 2 * theta / np.log(N)**3

    second_order = -m * (-an / bn) / 12 *(-6  * c2 * gamma**2 + gamma**3 *(6 * c2 **2 - 2 * c3) - c2 * np.pi**2 + gamma*(12 + (3 * c2**2-c3)*np.pi**2) + 12 * c2**2 * zeta(3) - 4 * c3 * zeta(3)) 

    return bn**m * (1 + second_order)