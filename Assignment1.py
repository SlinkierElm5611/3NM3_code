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
    # -P2 - sin(45)*P5 = 0
    # -P4 - sin(45)*P5 = 0
    # -P6 + sin(45)*P5 = 0
    # sin(45)*P3 + P4 -18 = 0
    # -sin(45)*P3 - P1 + P2 = 0

    # Part b
    A = np.array([
      [-1, 1, -1/np.sqrt(2), 0, 0, 0],
      [0, -1, 0, 0, -1/np.sqrt(2), 0],
      [0, 0, 1/np.sqrt(2), 1, 0, 0],
      [0, 0, 0, -1, -1/np.sqrt(2), 0],
      [0, 0, 0, 0, 1/np.sqrt(2), 0],
      [0, 0, 0, 0, 1/np.sqrt(2), -1]
    ])

    b = np.array([0, 0, 18, 0, 12, 0])
    P, L, U = sp.linalg.lu(A)
    y = sp.linalg.solve(L, b)
    x = sp.linalg.solve(U, y)
    print(f"Solution for P__i: {x}")
    # Part c
    b = np.array([0, 0, 36, 0, 24, 0])
    y = sp.linalg.solve(L, b)
    x = sp.linalg.solve(U, y)
    print(f"Solution for P__i with doubled loads: {x}")


# Problem 3
def problem3() -> None:
    """
    Given matrix A = [[1, 2], [3, 4]]
    a) Calculate the condition number of A
    b) Let's use a preconditioner matrix  P  to improve the condition number of the product  P−1A .
       Give 2 examples of  P  that improve the condition number, one of which being the 'perfect' preconditioner.
    """
    A = np.array([[1, 2], [3, 4]])
    # Part a
    condition_number = lambda x: np.linalg.norm(x)*np.linalg.norm(np.linalg.inv(x))
    print(f"Condition number of A: {condition_number(A)}")
    # Part b
    # Example 1: P = [[1, 0], [0, 1]]
    # This is the perfect preconditioner as it is the identity matrix.
    # Example 2: P = [[1, 0], [0, 0]]
    # This preconditioner will make the matrix singular and therefore improve the condition number.
    P__1 = np.array([[1, 0], [0, 1]])
    B = np.linalg.inv(P__1)@A
    print(f"Condition number of P__1A: {condition_number(B)}")
    P__2 = np.array([[1, 0], [0, 0]])
    B = np.linalg.inv(P__2)@A
    print(f"Condition number of P__2A: {condition_number(B)}")


if __name__ == "__main__":
    #problem1()
    problem2()
    #problem3()
