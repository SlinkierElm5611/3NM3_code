import numpy as np

def gauss_seidel(A, b, x0, tolerance=1e-4, max_iterations=1000):
    x = x0.copy()
    n = len(x)
    iteration = 1
    for i in range(max_iterations):
        x_prev = x.copy()
        for j in range(n):
            sum1 = sum(A[j][k] * x[k] for k in range(j))
            sum2 = sum(A[j][k] * x_prev[k] for k in range(j + 1, n))
            x[j] = (b[j] - sum1 - sum2) / A[j][j]
            iteration += 1
        if np.linalg.norm(x - x_prev) < tolerance:
            return x, i + 1
    return x, iteration

def Problem1():
    """
    Part a) Write out equilibrium equations for the positions of the blokcs and form a linear system.
    80    - x__1  * 2000 + (x__2 - x__1) * 3000 = 0
    (x__3 - x__2) * 3000 - (x__2 - x__1) * 3000 = 0
    (x__4 - x__3) * 3000 - (x__3 - x__2) * 3000 = 0
    (x__5 - x__4) * 3000 - (x__4 - x__3) * 3000 - 60 = 0
    (6    - x__5) * 2000 - (x__5 - x__4) * 3000 = 0
    """
    A = np.array([
        [-5000, 3000, 0, 0, 0],
        [3000, -6000, 3000, 0, 0],
        [0, 3000, -6000, 3000, 0],
        [0, 0, 3000, -6000, 3000],
        [0, 0, 0, 3000, -5000]
        ])
    b = np.array([-80, 0, 0, 60, -12000])
    """
    Part b) Use Gauss-Seidel iterations to solve this system. Use an absolute tolerance on the residual's Frobenius norm of 1e-4 to determine convergence. Report the solution and the number of iterations required.
    """
    x, iterations = gauss_seidel(A, b, np.zeros(5))
    print("Solution: ", x)
    print("Number of iterations: ", iterations)
    print("Numpy solution: ", np.linalg.solve(A, b))

def Problen2():
    """
    You have a new nuclear fuel type which is an infinite square bar, 1m in edge length. Its thermal conducitivity is  𝑘=2𝑊/𝑚2 . During irradiation, it generate fission heat  𝑄=1𝑘𝑊/𝑚3  and is cooled with heat pipes which keep the surface temperature exactly  100 0𝐶 .

    Reminder: The steady-state heat transport equation is:  −∇⋅[−𝑘∇𝑇]=−𝑄

    Part a) Discretize the problem with a 100x100 mesh and find the maximum temperature in steady state.
    """

if __name__ == "__main__":
    Problem1()
    Problen2()