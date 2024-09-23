import numpy as np


def Problem1():
    """
    Part a) Write out equilibrium equations for the positions of the blokcs and form a linear system.
    80    - x__1  * 2000 + (x__2 - x__1) * 3000 = 0
    (x__3 - x__2) * 3000 - (x__2 - x__1) * 3000 = 0
    (x__4 - x__3) * 3000 - (x__3 - x__2) * 3000 = 0
    (x__5 - x__4) * 3000 - (x__4 - x__3) * 3000 - 60 = 0
    (6    - x__5) * 3000 - (x__5 - x__4) * 3000 = 0
    """
    A = np.array([
        [-5000, 3000, 0, 0, 0],
        [3000, -6000, 3000, 0, 0],
        [0, 3000, -6000, 3000, 0],
        [0, 0, 3000, -6000, 3000],
        [0, 0, 0, 3000, -6000]
        ])
    b = np.array([-80, 0, 0, 60, -18000])
    x = np.linalg.solve(A, b)
    print(x)


if __name__ == "__main__":
    Problem1()
