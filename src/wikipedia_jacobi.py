import numpy as np


def jacobi(A, b, x_init, epsilon=1e-10, max_iterations=500):
    D = np.diag(np.diag(A))
    LU = A - D
    x = x_init
    D_inv = np.diag(1 / np.diag(D))
    for i in range(max_iterations):
        x_new = np.dot(D_inv, b - np.dot(LU, x))
        if np.linalg.norm(x_new - x) < epsilon:
            return x_new
        x = x_new
    return x

# problem data
A = np.array([
    [5, 2, 1, 1],
    [2, 6, 2, 1],
    [1, 2, 7, 1],
    [1, 1, 2, 8]
])
b = np.array([29, 31, 26, 19])

# you can choose any starting vector
x_init = np.zeros(len(b))
x = jacobi(A, b, x_init)

print("x:", x)
print("computed b:", np.dot(A, x))
print("real b:", b)