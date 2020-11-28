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
        [5, 2, 1, 1],
        [2, 6, 2, 1],
        [1, 2, 7, 1],
        [1, 1, 2, 8]
    ]
)
test_result = np.array(
    [29, 31, 26, 19]
)
test_guess = np.array(
    [0., 0., 0., 0.]
)
actual_x = np.array(
    [551./138, 1223./414, 895./414, 200./207]
)


# checks if pivot i in the given line is dominant.
def check_dominant_pivot(tested_line, i):
    local_sum = 0
    # sum up all of the members of the line
    for j in range(i):
        local_sum += abs(tested_line[j])
    # test if the pivot is greater than the sum of the other members
    if tested_line[i] >= local_sum - tested_line[i]:
        return True
    return False


# checks if the diagonal in the given matrix is dominant.
def check_dominant_diagonal(test_matrix):
    for i in range(test_matrix.shape[0]):
        # check if pivot i is dominant
        if not check_dominant_pivot(test_matrix[i], i):
            return False
    # all of the pivots are dominant, the diagonal is dominant
    return True


def test_dominant_diagonal():
    print("matrix: ", test_mat)
    print("is matrix dominant diagonally?:", check_dominant_diagonal(test_mat))


def jacobi_method(mat, result, guess, error=epsilon, max_iterations=max_iter):
    diagonal = np.diag(np.diag(mat))
    lower_upper = mat - diagonal
    dinverse = np.linalg.inv(diagonal)
    for j in range(max_iterations):
        print("{0}th guess: {1}".format(j, str(guess)))

        # the stuff the algorithm does to get the next guess
        new_guess = np.dot(dinverse, result - np.dot(lower_upper, guess))

        # end clause, once guesses are really close between iterations
        norm = np.linalg.norm(guess - new_guess)
        if norm <= error:
            return new_guess

        # update guess to be the one we just found
        guess = new_guess.copy()
    return guess


def test_jacobi():
    x = jacobi_method(test_mat, test_result, test_guess)
    print("matrix:\n", test_mat)
    print("guessed b:\n", np.dot(test_mat, x))
    print("actual b:\n", test_result)


def gauss_seidel_method(mat, result, guess, error=epsilon, max_iterations=max_iter):
    diagonal = np.diag(np.diag(mat))
    lower = np.tril(mat) - diagonal
    upper = mat - np.tril(mat)
    lower_diagonal = diagonal + lower
    lower_diagonal_inv = np.linalg.inv(lower_diagonal)
    for j in range(0, max_iterations + 1):
        print("{0}th guess: {1}".format(j, str(guess)))

        # the stuff the algorithm does to get the next guess
        new_guess = np.dot(lower_diagonal_inv, result - np.dot(upper, guess))

        # end clause, once guesses are really close between iterations
        norm = np.linalg.norm(guess - new_guess)
        if norm <= error:
            return new_guess

        # end clause was not reached, update guess to be the new guess and try again
        guess = new_guess.copy()
    return guess


def test_gauss_seidel():
    x = gauss_seidel_method(test_mat, test_result, test_guess)
    print("matrix:\n", test_mat)
    print("guessed b:\n", np.dot(test_mat, x))
    print("actual b:\n", test_result)


def main():
    test_jacobi()
    test_gauss_seidel()