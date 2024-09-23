import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import gmres, splu, LinearOperator
import matplotlib.pyplot as plt

def Problem1():
    """
    Assuming that the distance from left end of system to right end is 6M
    Part a) Write out equilibrium equations for the positions of the blokcs and form a linear system.
    80    - x__1  * 2000 + (x__2 - x__1) * 3000 = 0
    (x__3 - x__2) * 3000 - (x__2 - x__1) * 3000 = 0
    (x__4 - x__3) * 3000 - (x__3 - x__2) * 3000 = 0
    (x__5 - x__4) * 3000 - (x__4 - x__3) * 3000 - 60 = 0
    (0    - x__5) * 2000 - (x__5 - x__4) * 3000 = 0
    """
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

    A = np.array([
        [-5000, 3000, 0, 0, 0],
        [3000, -6000, 3000, 0, 0],
        [0, 3000, -6000, 3000, 0],
        [0, 0, 3000, -6000, 3000],
        [0, 0, 0, 3000, -5000]
        ])
    b = np.array([-80, 0, 0, 60, 0])
    """
    Part b) Use Gauss-Seidel iterations to solve this system. Use an absolute tolerance on the residual's Frobenius norm of 1e-4 to determine convergence. Report the solution and the number of iterations required.
    """
    x, iterations = gauss_seidel(A, b, np.zeros(5))
    print("Solution: ", x)
    print("Number of iterations: ", iterations)
    print("Numpy solution: ", np.linalg.solve(A, b))

def Problem2():
    """
    You have a new nuclear fuel type which is an infinite square bar, 1m in edge length. Its thermal conducitivity is  𝑘=2𝑊/𝑚2 . During irradiation, it generate fission heat  𝑄=1𝑘𝑊/𝑚3  and is cooled with heat pipes which keep the surface temperature exactly  100 0𝐶 .

    Reminder: The steady-state heat transport equation is:  −∇⋅[−𝑘∇𝑇]=−𝑄

    Part a) Discretize the problem with a 100x100 mesh and find the maximum
    temperature in steady state.
    Part b) Ideally, your solution will be independant of your mesh size.
    Conduct a mesh sensativity analysis (which means to halve the mesh size and
    rerun) to check if your solution depend on your discretization.
    Use an iterative method and comment on the effectiveness of an ILU
    preconditioner.
    """
    def computeTemperature(n):
        L = 1.0  # length of the square bar (1 meter)
        k = 2.0  # thermal conductivity (W/m^2)
        Q = 1000.0  # heat generation (W/m^3)
        T_boundary = 100.0  # boundary temperature (°C)

        dx = L / (n - 1)

        N = n * n

        A = lil_matrix((N, N))
        B = np.zeros(N)

        def index(i, j):
            return i * n + j

        for i in range(n):
            for j in range(n):
                idx = index(i, j)

                if i == 0 or i == n-1 or j == 0 or j == n-1:
                    A[idx, idx] = 1
                    B[idx] = T_boundary
                else:
                    A[idx, index(i+1, j)] = 1  # T(i+1, j)
                    A[idx, index(i-1, j)] = 1  # T(i-1, j)
                    A[idx, index(i, j+1)] = 1  # T(i, j+1)
                    A[idx, index(i, j-1)] = 1  # T(i, j-1)
                    A[idx, idx] = -4  # T(i, j)

                    B[idx] = -Q * dx**2 / k

        A = A.tocsr()

        lu = splu(A)

        M_x = LinearOperator(A.shape, matvec=lu.solve)

        T, exitCode = gmres(A, B, M=M_x, atol=1e-6, rtol=1e-6)

        if exitCode == 0:
            print("GMRES converged successfully with LU preconditioner.")
        else:
            print(f"GMRES failed to converge. Exit code: {exitCode}")

        T_grid = T.reshape((n, n))

        max_temperature = np.max(T_grid)
        print(f"Maximum temperature in the system: {max_temperature:.2f} °C")

        plt.imshow(T_grid, cmap='hot', interpolation='nearest')
        plt.colorbar(label="Temperature (°C)")
        plt.title("Temperature Distribution in the Nuclear Fuel Square Bar")
        plt.show()
    computeTemperature(100) #Part a
    computeTemperature(50) #Part b half mesh check
    #Part A and part b show the same results even though 

if __name__ == "__main__":
    Problem1()
    Problem2()
