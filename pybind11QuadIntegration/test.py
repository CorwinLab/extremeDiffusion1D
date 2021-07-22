import numpy as np
import npquad
import quadTest

a = (quadTest.return_vector())
b = np.array([5, 6, 7], dtype=np.float64)
print(b)
c = quadTest.double_vector(b)
print(type(c[0]))
c = np.array(c)
print(type(c[0]))
print(type(c))
print(c.dtype)

print(type(a))
print(type(a[0]))
