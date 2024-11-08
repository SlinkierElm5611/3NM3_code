import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
def Problem1():
    """
    You are calculating the temperature profile in a nuclear plate fuel that is 1mm thick; much thinner than
    the other dimensions. In a 1D model through the thickness, the temperature obeys
    the Fourier heat balance law with a heat source; grad*lambda * grad(T) = q
    The boundary conditions are that T=300K on all outer surfaces.
    a) Write the finite difference scheme for Fourier heat balance equation.
    T__0 = 300
    T__n-1 = 300
    T__i+1 - 2T__i + T__i-1 = -h^2 * q / lambda
    This becomes the following matrix equation:
    """
    def generateA(n):
        A = np.zeros((n+1, n+1))
        A[0, 0] = 1
        A[n, n] = 1
        for i in range(1,n):
            A[i, i] = -2
            A[i, i-1] = 1
            A[i, i+1] = 1
        return A
    def generateb(n, h, q, lam, boundary=300):
        b = np.zeros(n+1)
        b[0] = boundary
        b[1:-1] = -h**2 * q / lam
        b[-1] = boundary
        return b
    d__0 = 0
    d__1 = 1e-3
    Q = 1e3
    lam = 2e-3
    h = 10e-6
    n = 100
    boundary = 300
    A = generateA(n)
    b = generateb(n, h, Q, lam, boundary)
    T = np.linalg.solve(A, b)
    d = np.linspace(d__0, d__1, n+1)
    plt.plot(d, T)
    plt.show()

    """
    c) Write the Finite Difference formula if lambda = 2 + T/300
    d/dx*(lambda*dT/dx)  = -q
    d/dx*(2+T/300*dT/dx) = -q
    1/300*dT/dx + d^2T/dx^2 = -q
    (T__i+1 - T__i-1) / 2h + (T__i+1 - 2T__i + T__i-1) / h^2 = -q
    T__i+1^2 - 2T__i*T__i+1 + 2T__i*T__i-1 - T__i-1^2 + 600h^3*q = 0
    R(T) = 0
    """
    #d) Solve for T with the temperature dependent thermal conductivity

def Problem2():
    """
    a)
    Write a function that takes a  2−π  periodic function and a degree  n , then outputs the  n−th  Fourier coefficients  [An,Bn] . Use the equations above with a suitable integration method. (Don't use packaged Fourier analysis tools)

    Test it for a suitable set of functions for which you have an analytic answer.
    """
    def FourierCoefficients(f, n):
        A = np.zeros(n)
        B = np.zeros(n)
        for i in range(1, n+1):
            A[i-1] = (1 / np.pi) * sp.integrate.quad(lambda x: f(x) * np.cos(i * x), -np.pi, np.pi)[0]
            B[i-1] = (1 / np.pi) * sp.integrate.quad(lambda x: f(x) * np.sin(i * x), -np.pi, np.pi)[0]
        return A, B
    # test
    print("Test")
    def f(x):
        return np.sin(x)
    n = 5
    A, B = FourierCoefficients(f, n)
    print("A: ", A)
    print("B: ", B)
    def f(x):
        return np.cos(x)
    n = 5
    A, B = FourierCoefficients(f, n)
    print("A: ", A)
    print("B: ", B)
    def f(x):
        return np.sin(x) + np.cos(x)
    n = 5
    A, B = FourierCoefficients(f, n)
    print("A: ", A)
    print("B: ", B)
    print("End Test")
    """
    b) Find the cooefficients for the following functions
    """
    functions = [[lambda x: np.mod(x, np.pi/2), 5],
                 [lambda x: np.mod(x, np.pi/2), 20],
                 [lambda x: (x > -np.pi/2) & (x < np.pi/2), 2],
                 [lambda x: (x > -np.pi/2) & (x < np.pi/2), 20]]
    for f, n in functions:
        print("Function: ", f)
        print("n: ", n)
        A, B = FourierCoefficients(f, n)
        print("A: ", A)
        print("B: ", B)

def Problem3():
    """
    Given a cubic function f(x) with:
        integral from -1 to 1 of f(x)dx = 3
        and f(-3^(-1/2)) = 1
    what is the value of f(3^(-1/2))?
    Why?
    Using the 2-point Gauss-Legendre formula: I = c1*f(x1) + c2*f(x2)
    I = c1*f(x1) + c2*f(x2) = integral from -1 to 1 of f(x)dx
    I = c1*f(x1) + c2*f(x2) = 3
    c1*f(x1) = f(-3^(-1/2))
    therefore f(3^(-1/2)) = 3-f(-3^(-1/2)) = 3-1 = 2
    """
    pass

if __name__ == "__main__":
    Problem1()
    Problem2()
