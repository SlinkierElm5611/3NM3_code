import numpy as np
import scipy as sp

def Problem1():
    L = 160
    h = 15

    def f(x):
        #part a)
        # x is a column vector of [lambda, s]
        lambda_val, s = x
        #part b)
        # the functions to solve are the column vector below being returned
        return np.array([
            h - (1 / lambda_val) * (np.cosh(lambda_val * L / 2) - 1),
            s - (2 / lambda_val) * np.sinh(lambda_val * L / 2)
        ])

    def jacobian(x):
        lambda_val, s = x
        df1_dlambda = (1 / lambda_val**2) * (np.cosh(lambda_val * L / 2) - 1) - (L / (2 * lambda_val)) * np.sinh(lambda_val * L / 2)
        df2_dlambda = (2 / lambda_val**2) * np.sinh(lambda_val * L / 2) - (L / lambda_val) * np.cosh(lambda_val * L / 2)

        return np.array([
            [df1_dlambda, 0],  # df1/ds is always 0
            [df2_dlambda, 1]   # df2/ds is always 1
        ])

    def newton_raphson(x__0, tolerance=1e-6, max_iterations=100):
        x = x__0
        for iteration in range(max_iterations):
            f_x = f(x)
            residual = np.linalg.norm(f_x)

            print(f"Iteration {iteration + 1}: Current guess = {np.round(x, 6)}, Residual = {residual:.6f}")

            if residual < tolerance:
                print(f"Converged after {iteration + 1} iterations.")
                return x

            J_x = jacobian(x)
            delta_x = sp.linalg.solve(J_x, -f_x)

            x = x + delta_x

        print("No solution found within the maximum number of iterations.")

    #initial guess
    x__0 = np.array([0.01,L])
    solution = newton_raphson(x__0)

    if solution is not None:
        lambda_val, s = solution
        print(f"lambda = {lambda_val:.6f}, s = {s:.6f}")
    else:
        print("No solution found.")

    """
    d)
    I chose to use the Newton Raphson method because it iterates through creating linear systems to create the next best guess.
    This method uses the last two guesses to create a linear system and calculate the gradient of that system.
    This allows us to essentially "roll down the hill" to a right answer as we iterate.
    """

def Problem2():
    def f(x):
        x, y = x
        #part a)
        return np.array([
            (x - 2)**2 + y**2 - 4,
            x**2 + (y - 3)**2 - 4
        ])
    
    def jacobian(x):
        x, y = x
        #part b)
        return np.array([
            [2 * (x - 2), 2 * y],
            [2* x, 2 * (y - 3)]
        ])

    def newton_raphson(x__0, tol=1e-6, max_iter=100):
        x = x__0
        for _ in range(max_iter):
            f_x = f(x)
            if np.linalg.norm(f_x) < tol:
                return x

            # Compute Jacobian and update the solution
            J_x = jacobian(x)
            delta_x = sp.linalg.solve(J_x, -f_x)
            x = x + delta_x

    def find_two_solutions(guesses, tol=1e-6, max_iter=100):
        solutions = []
        for guess in guesses:
            solution = newton_raphson(guess, tol, max_iter)
            if solution is not None and all(np.linalg.norm(solution - sol) > tol for sol in solutions):
                solutions.append(solution)
            if len(solutions) == 2:
                break
        return solutions

    initial_guesses = [
        np.array([0.01, 1]),
        np.array([-1, 3]),
        np.array([3, 3]),
        np.array([4, 4]),
        np.array([5, 5])
    ]

    distinct_solutions = find_two_solutions(initial_guesses)

    if len(distinct_solutions) == 2:
        print(f"Two distinct solutions found: {distinct_solutions}")
        x__1, y__1 = distinct_solutions[0]
        x__2, y__2 = distinct_solutions[1]
        print(f"x__1 = {x__1:.6f}, y__1 = {y__1:.6f}")
        print(f"x__2 = {x__2:.6f}, y__2 = {y__2:.6f}")
    else:
        print("No two distinct solutions found.")

if __name__ == "__main__":
    Problem1()
    Problem2()
