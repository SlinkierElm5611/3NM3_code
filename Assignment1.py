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
    significantly slower performance at lower N values.
    """


if __name__ == "__main__":
    problem1()
