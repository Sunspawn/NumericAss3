"""Nikolay Babkin, 321123242

Assignment 3

Write 2 functions solving a system of linear equations using the Jacobi and Gauss-Seidel methods.

Every vector and matrix is a numpy array object.
"""

import numpy as np

epsilon = 1e-10
max_iter = 500
test_mat = np.array(
    [
        (2, 1),
        (1, -2)
    ]
)
test_result = np.array(
    [2, -2]
)
test_guess = np.array(
    [0, 0]
)


# checks if pivot i in the given line is dominant.
def check_dominant_pivot(tested_line, i):
    local_sum = 0
    # sum up all of the members of the line except the pivot
    for j in range(0, i):
        local_sum += abs(tested_line[j])
    for j in range(i+1, tested_line.size):
        local_sum += abs(tested_line[j])
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
    print("matrix: \n" + str(test_mat))
    print("1st pivot check: \n" + str(check_dominant_pivot(test_mat[1], 1)))
    print("full test \n" + str(check_dominant_diagonal(test_mat)))


# Jacobi method

# finds the ith value of the guess
def find_xi(mat, result, guess, i):
    line = mat[i]
    bi = result[i]
    xi = result[i] * 1.0
    for j in range(0, i):
        xi -= guess[j] * mat[i][j]
    for j in range(i+1, line.size):
        xi -= guess[j] * mat[i][j]
    return xi / mat[i][i]


def test_find_xi():
    for i in range(0, test_result.size):
        print(str(find_xi(test_mat, test_result, test_guess, i)))


def jacobi_method(mat, result, guess, error=epsilon, max_iterations=max_iter):
    for j in range(0, max_iterations + 1):
        new_guess = np.array(guess).astype('float64')

        for i in range(0, guess.size):
            new_guess[i] = find_xi(mat, result, guess, i)

        # end clause, once guesses are really close between iterations
        if np.linalg.norm(guess - new_guess, np.inf) <= error:
            j = max_iterations + 1

        # end clause was not reached, update guess to be the new guess and try again
        guess = new_guess.copy()
    return guess


def test_jacobi():
    print("matrix: \n" + str(test_mat))
    print("result: \n" + str(test_result))
    print("guess: \n" + str(test_guess))
    print("final guess: \n" + str(jacobi_method(test_mat, test_result, test_guess)))
