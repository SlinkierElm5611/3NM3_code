from typing import List
import numpy as np
import scipy as sp
import sympy as sym
import matplotlib.pyplot as plt
import timeit

"""
Important note.
While Dr.Welland has allowed us to submit our code as a .py file,
the magic command %timeit is not supported in a .py file.
Therefore, the equivalent module timeit is being used to approximate the time complexity.
"""


# Problem 1
def problem1() -> None:
    """
    Examine the value of sparse matricies by comparing the approximate computational
    efficiencies of sparse and dense solvers on a tridiagonal system.
    Use the %timeit function to approximate the complexity.
    """
    sizes = np.array([2**i for i in range(1, 14)])
    time_ratio: List[float] = []
    for i in range(len(sizes)):
        main_diag = np.full(sizes[i], 2)
        upper_diag = np.full(sizes[i] - 1, -1)
        lower_diag = np.full(sizes[i] - 1, -1)
        A_sparce = sp.sparse.diags([main_diag, upper_diag, lower_diag], [0, 1, -1], format='csr')
        A_dense = A_sparce.toarray()
        b = np.random.rand(sizes[i])
        x_sparce_duration = timeit.timeit(lambda: sp.sparse.linalg.spsolve(A_sparce, b), number=1)

        x_dense_duration = timeit.timeit(lambda: np.linalg.solve(A_dense, b), number=1)

        print(f"For matrix size {sizes[i]}:")
        print(f"Time for sparse solver: {x_sparce_duration}")
        print(f"Time for dense solver: {x_dense_duration}")
        time_ratio.append(x_dense_duration/x_sparce_duration)
    plt.plot(sizes, time_ratio)
    plt.xlabel("Matrix Size")
    plt.ylabel("Time Ratio (Dense/Sparse)")
    plt.title("Time Ratio vs Matrix Size")
    plt.show()
    """
    What can be observed from the resulting graph and the printed time values is that
    the sparse solver is significantly less computationally expensive than the dense solver
    for large values of N. However, it has a higher constant cost for very small values of N resulting in 
    significantly slower performance at lower N values. This behaviour is similar to what is seen
    in hash tables, where the constant cost of hashing the key is on average higher than the time it takes to
    search through a list of small N values(usually 7 in the case of hash tables and stack array iteration).
    """


# Problem 2
def problem2() -> None:
    """
    a) Write the linear system for P__i
    b) Solve for P__i using decomposition and back substitution
    c) Double the loads (18 kN and 12 kN) and solve for P__i again without refactoring the matrix
    """
    # Part a
    # sin(45)*P5 -12 = 0
    # P2 + sin(45)*P5 = 0
    # P4 + sin(45)*P5 = 0
    # P6 + sin(45)*P5 = 0
    # sin(45)*P3 + P4 -18 = 0
    # sin(45)*P3 + P1 + P2 = 0

    # Part b
    A = np.array([[1, 1, np.sin(np.pi/4), 0, 0, 0],
                  [0, 1, 0, 0, np.sin(np.pi/4), 0],
                  [0, 0, np.sin(np.pi/4), 1, 0, 0],
                  [0, 0, 0, 1, np.sin(np.pi/4), 0],
                  [0, 0, 0, 0, np.sin(np.pi/4), 0],
                  [0, 0, 0, 0, np.sin(np.pi/4), 1]])
    b = np.array([0, 0, 18, 0, 12, 0])
    L = np.zeros((6, 6))
    np.fill_diagonal(L, 1)
    # U is initialized as a copy of A to allow us to write inplace and make algorithm more efficient
    U = A.copy()
    for i in range(A.diagonal().size):
        for j in range(i+1, A.diagonal().size):
            L[j, i] = U[j, i]/U[i, i]
            U[j] -= L[j, i]*U[i]
    print(f"L matrix: {L}")
    print(f"U matrix: {U}")


if __name__ == "__main__":
    problem2()
