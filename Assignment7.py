import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
def Problem1():
    """
    The Blasius equation is a Boundary Value Problem that appears in fluid mechanics as a laminar flow boundary layer and is written
    """
    # a) Express this 3rd order ODE as a system of 3 first order ODEs
    # y1,
    # y2 = y1',
    # y3 = y2'
    # y''' + y*y'' = 0 --> y3' + y1*y3 = 0

    # The three equations are:
    # y1' = y2
    # y2' = y3
    # y3' = -y1*y3
    # Define the Blasius system of ODEs
    # b)
    def blasiusSystem(t, y):
        y1, y2, y3 = y
        return [y2, y3, -y1 * y3]

    targetValue = 2

    tSpan = (0, 10)
    tEval = np.linspace(0, 10, 500)

    y1_0 = 0
    y2_0 = 0

    def shooting_function(y3_initial_guess):
        yInitial = [y1_0, y2_0, y3_initial_guess[0]]
        sol = sp.integrate.solve_ivp(blasiusSystem, tSpan, yInitial, t_eval=tEval, method='RK45')
        return sol.y[1][-1] - targetValue

    initial_guess = [0.3]
    result = sp.optimize.root(shooting_function, initial_guess)

    y3_0 = result.x[0]

    yInitial = [y1_0, y2_0, y3_0]
    solution = sp.integrate.solve_ivp(blasiusSystem, tSpan, yInitial, t_eval=tEval, method='RK45')

    plt.figure(figsize=(10, 6))
    plt.plot(solution.t, solution.y[0], label='y(x)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Solution using the Shooting Method')
    plt.grid(True)
    plt.legend()
    plt.show()

    print(solution)

def Problem2():
    """
    a)
    The implicit Euler method can be used as a timestepping scheme. For a steady state heat system, implicit euler is very appropriate and stable for this use case.
    when discretized, it looks like this:
        Ti^t+1 = Ti^t + [Ti-1^t+1 - 2Ti^t+1 + Ti+1^t+1]
    b) Code your scheme and determine when you will reach steady state to within 1%
    """
    alpha = 1.0
    rMin = 0.5
    rMax = 1.0
    Nr = 100
    dt = 0.005
    tolerance = 0.01
    maxSteps = 10000

    r = np.linspace(rMin, rMax, Nr)
    dr = (rMax - rMin) / (Nr - 1)

    T = np.ones(Nr) * 200
    T[0] = 0

    A = np.zeros((Nr, Nr))
    coeff = alpha * dt / (dr**2)

    for i in range(1, Nr - 1):
        r_factor = alpha * dt / (2 * r[i] * dr)
        A[i, i - 1] = -coeff + r_factor
        A[i, i] = 1 + 2 * coeff
        A[i, i + 1] = -coeff - r_factor

    A[0, 0] = 1
    A[-1, -1] = 1

    T_new = T.copy()
    for step in range(maxSteps):
        T_new = np.linalg.solve(A, T)
        T_diff = np.linalg.norm(T_new - T) / np.linalg.norm(T_new)
        T[:] = T_new
        if T_diff < tolerance:
            break

    plt.figure(figsize=(8, 5))
    plt.plot(r, T, label=f'Steady-state (t={step * dt:.3f}s)', color='green')
    plt.xlabel('Radius (r)')
    plt.ylabel('Temperature (°C)')
    plt.title('Steady-State Temperature Profile (Implicit Euler)')
    plt.grid()
    plt.legend()
    plt.show()

    print(f"Steady state reached in {step} steps, time = {step * dt:.3f} seconds.")

if __name__ == "__main__":
    Problem1()
    Problem2()
