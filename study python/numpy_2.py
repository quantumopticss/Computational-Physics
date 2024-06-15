import numpy as np
import cmath

A = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])

A[1:3,1:3] = [[1,1],[2,2]]
b = (A[1:3,[0,1,2]])[0,[1,2]] # [[4,5,6],[7,8,9]] -> [5,6]
c = A[...,[1,2]]

print(A);print(b),print(c)

theta = 2*np.pi*np.random.rand(5)
print(theta)

a = np.array([0,1,2])
print(cmath.exp(1j*a))