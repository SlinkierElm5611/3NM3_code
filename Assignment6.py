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
    T__i+1 - 2T__i + T__i-1 = -q * dx**2 / lambda
    b) Assuming q = 1kW/m^3, lambda = 2 mW/mK, and dx = 10um, use the finite difference to find a solution.
    """
    # b)
    q = 1e3 # W/m^3
    lambda_ = 2e-3 # mW/mK
    dx = 10e-6 # m
    n = 1e-3 / dx
    T = np.zeros(n)
    T[0] = 300
    T[-1] = 300
    for i in range(1, n-1):
        T[i] = (T[i+1] + T[i-1] + q * dx**2 / lambda_) / 2
    plt.plot(T)
    plt.show()

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

if __name__ == "__main__":
    #Problem1()
    Problem2()
