from hack_itau_quant.optimization.quadratic_programming import QuadraticProgrammig
import numpy as np

B = np.array([[1, 2], [3, 4]])
A = np.array([[1, 1]])
c = np.array([[1]])

qp = QuadraticProgrammig(B = B, A = A, c = c)

w, l = qp.solve()

print("---------------- MATRIZ W:")
print(w)
print("---------------- MATRIZ lambda:")
print(l)


#print(np.linalg.inv(C) @ d)
