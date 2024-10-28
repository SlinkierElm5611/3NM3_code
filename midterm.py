import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

def Problem1():
    """
    You havea a series of blocks connected by springs. You apply a force as shown below;
    -5000*dx1 - 3000*dx2 = -5000
     3000*dx1 - 6000*dx2  = 0
    """
    A = np.array([[-5000, -3000], [3000, -6000]])
    b = np.array([-5000, 0])
    x = np.linalg.solve(A, b)
    print("Displacement of blocks: ", x)

def Problem2():
    """
    Use a minimizer to find the minimum of:
        f(x) = (x-2)^2 - 2cos(5x)
    """
    def f(x):
        return (x-2)**2 - 2*np.cos(5*x)
    x = np.linspace(-10, 10, 100)
    y = f(x)
    # found initial guess by plotting function and looking for area near minimum
    x0 = 2.5
    res = sp.optimize.minimize(f, x0)
    print(" Optimal x: ", res.x)

def Problem3():
    """
    TRISO2 (TRistructural ISOstropic 2) nuclear fuel is a spherical fuel design with a radius of 10mm.
    The boundary of the sphere is fixed at 300K.

    You have data from a temperature probe in | x | y | z | T | format below.
    """
    data = np.array([[-2.41689854e+00,  4.84339485e+00, -6.16870630e+00,
                      3.27795379e+02],
                     [ 6.37459929e-01, -2.31322296e+00, -5.12925643e+00,
                      3.62979958e+02],
                     [-1.36585141e+00,  3.78414715e+00,  7.55713647e+00,
                      3.22407378e+02],
                     [-1.28436044e+00,  1.56279514e+00,  2.87315288e+00,
                      3.85150405e+02],
                     [-8.39743148e+00,  3.66068150e+00,  1.25910049e-01,
                      3.13132895e+02],
                     [ 6.01955390e-01, -1.33605143e+00,  9.72628945e+00,
                      3.02574868e+02],
                     [-1.45056695e+00, -8.95709338e+00, -2.97283166e+00,
                      3.07088218e+02],
                     [ 4.28093553e+00, -2.62182086e+00, -1.13247003e+00,
                      3.69068889e+02],
                     [-8.78201296e+00, -3.30560349e+00, -1.05490010e-01,
                      3.09658982e+02],
                     [ 2.72659396e-01, -5.15372488e-01, -9.32487868e+00,
                      3.10300430e+02]])
    """
    a) What symmetries are present?
    There is rotational symmetry around all axes.
    b) Based on the symmetry, what are the two boundary conditions for this problem?
    The boundary conditions are that the temperature is constant at the boundary of the sphere and that the temperature is symmetric around all axes.
    """
    #c) fit a quadratic to this data and plot it
    # making plots with respect to r as it is symmetrical

    r = np.sqrt(data[:, 0]**2 + data[:, 1]**2 + data[:, 2]**2)
    T = data[:, 3]
    def quadratic(x, a, b, c):
        return a*x**2 + b*x + c
    popt, pcov = sp.optimize.curve_fit(quadratic, r, T)
    plt.scatter(r, T)
    t = np.linspace(0, 10, 100)
    plt.plot(t, quadratic(t, *popt))
    plt.show()

if __name__ == '__main__':
    Problem1()
    Problem2()
    Problem3()
