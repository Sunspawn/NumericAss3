"""Nikolay Babkin, 321123242

Assignment 3

Write 2 functions solving a system of linear equations using the Jacobi and Gauss-Seidel methods.
"""

import numpy as np


# checks if pivot i in the given line is dominant.
def check_dominant_pivot(tested_line, i):
    local_sum = 0
    # sum up all of the members of the line except the pivot
    for j in range(0, i):
        local_sum += tested_line[j]
    for j in range(i+1, tested_line.size):
        local_sum += tested_line[j]
    # test if the pivot is greater than the sum of the other members
    if tested_line[i] >= local_sum:
        return True
    return False


# checks if the diagonal in the given matrix is dominant.
def check_dominant_diagonal(test_matrix):
    for i in range(0, test_matrix.shape[0]):
        # check if pivot i is dominant
        if not check_dominant_pivot(test_matrix[i], i):
            return False
    # all of the pivots are dominant, the diagonal is dominant
    return True


def test_dominant_diagonal():
    # testing
    a = np.array([(5, 2, 1), (2, 4, 1), (3, 1, 6)])
    print(check_dominant_pivot(a[1], 1))
    print(check_dominant_diagonal(a))
