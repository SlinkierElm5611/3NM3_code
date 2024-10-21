"""
Copilot use note, in-editor copilot suggestions used to complete lines intermittently throughout the code.
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sympy

def Problem1():
    # Given data in (T, k) columns, fit a modified Arrhenius curve of the form: k = (A+BT)*exp(-E/RT)
    data = np.array([[2.73000000e+02, 4.45900132e+03],
           [2.76030303e+02, 4.41776331e+03],
           [2.79060606e+02, 7.02912764e+03],
           [2.82090909e+02, 8.07233255e+03],
           [2.85121212e+02, 1.35655353e+04],
           [2.88151515e+02, 1.53935301e+04],
           [2.91181818e+02, 1.66518521e+04],
           [2.94212121e+02, 2.57056205e+04],
           [2.97242424e+02, 2.80402378e+04],
           [3.00272727e+02, 3.52850720e+04],
           [3.03303030e+02, 2.53348206e+04],
           [3.06333333e+02, 6.20422294e+04],
           [3.09363636e+02, 5.65720969e+04],
           [3.12393939e+02, 8.94469023e+04],
           [3.15424242e+02, 9.65941942e+04],
           [3.18454545e+02, 1.14139663e+05],
           [3.21484848e+02, 1.00384893e+05],
           [3.24515152e+02, 1.60315046e+05],
           [3.27545455e+02, 1.75487522e+05],
           [3.30575758e+02, 2.79200552e+05],
           [3.33606061e+02, 2.83492579e+05],
           [3.36636364e+02, 2.19343286e+05],
           [3.39666667e+02, 3.46339717e+05],
           [3.42696970e+02, 3.29907107e+05],
           [3.45727273e+02, 2.99334626e+05],
           [3.48757576e+02, 4.85091915e+05],
           [3.51787879e+02, 7.28675595e+05],
           [3.54818182e+02, 9.91612106e+05],
           [3.57848485e+02, 9.39914377e+05],
           [3.60878788e+02, 1.46925677e+06],
           [3.63909091e+02, 1.46126057e+06],
           [3.66939394e+02, 1.35738053e+06],
           [3.69969697e+02, 1.14736966e+06],
           [3.73000000e+02, 2.10254656e+06],
           [3.76030303e+02, 1.12017395e+06],
           [3.79060606e+02, 2.34226319e+06],
           [3.82090909e+02, 2.22858410e+06],
           [3.85121212e+02, 3.26613076e+06],
           [3.88151515e+02, 3.68055559e+06],
           [3.91181818e+02, 2.37869512e+06],
           [3.94212121e+02, 4.10021974e+06],
           [3.97242424e+02, 3.84314791e+06],
           [4.00272727e+02, 6.46893402e+06],
           [4.03303030e+02, 5.35422811e+06],
           [4.06333333e+02, 7.73962115e+06],
           [4.09363636e+02, 8.41409713e+06],
           [4.12393939e+02, 7.81006766e+06],
           [4.15424242e+02, 8.35579617e+06],
           [4.18454545e+02, 1.03669786e+07],
           [4.21484848e+02, 1.02992887e+07],
           [4.24515152e+02, 6.39189417e+06],
           [4.27545455e+02, 1.36405449e+07],
           [4.30575758e+02, 1.49410729e+07],
           [4.33606061e+02, 2.09149077e+07],
           [4.36636364e+02, 2.14703032e+07],
           [4.39666667e+02, 2.11464887e+07],
           [4.42696970e+02, 2.60359855e+07],
           [4.45727273e+02, 3.19593647e+07],
           [4.48757576e+02, 2.84547486e+07],
           [4.51787879e+02, 2.98421720e+07],
           [4.54818182e+02, 3.54693846e+07],
           [4.57848485e+02, 4.17941703e+07],
           [4.60878788e+02, 4.65948030e+07],
           [4.63909091e+02, 5.74740135e+07],
           [4.66939394e+02, 4.12282427e+07],
           [4.69969697e+02, 6.43295352e+07],
           [4.73000000e+02, 6.78921367e+07],
           [4.76030303e+02, 6.87155479e+07],
           [4.79060606e+02, 6.70678776e+07],
           [4.82090909e+02, 6.23342574e+07],
           [4.85121212e+02, 9.48617775e+07],
           [4.88151515e+02, 9.34710486e+07],
           [4.91181818e+02, 8.25781385e+07],
           [4.94212121e+02, 1.02888832e+08],
           [4.97242424e+02, 9.99071191e+07],
           [5.00272727e+02, 1.01319698e+08],
           [5.03303030e+02, 1.32602108e+08],
           [5.06333333e+02, 1.31923942e+08],
           [5.09363636e+02, 1.00146518e+08],
           [5.12393939e+02, 1.48398130e+08],
           [5.15424242e+02, 2.01128065e+08],
           [5.18454545e+02, 2.25792703e+08],
           [5.21484848e+02, 1.99863590e+08],
           [5.24515152e+02, 2.47074414e+08],
           [5.27545455e+02, 2.08548654e+08],
           [5.30575758e+02, 1.83449792e+08],
           [5.33606061e+02, 2.58856094e+08],
           [5.36636364e+02, 3.10038100e+08],
           [5.39666667e+02, 3.58956357e+08],
           [5.42696970e+02, 3.21512830e+08],
           [5.45727273e+02, 4.30773903e+08],
           [5.48757576e+02, 4.39179856e+08],
           [5.51787879e+02, 4.57110476e+08],
           [5.54818182e+02, 3.75292342e+08],
           [5.57848485e+02, 3.97574548e+08],
           [5.60878788e+02, 4.57516936e+08],
           [5.63909091e+02, 2.63845055e+08],
           [5.66939394e+02, 4.59249219e+08],
           [5.69969697e+02, 5.23677345e+08],
           [5.73000000e+02, 7.51098462e+08]])
    T = data[:,0]
    k = data[:,1]
    R = 8.314
    def modified_arrhenius(T, A, B, E):
        return (A + B*T)*np.exp(-E/(R*T))
    initial_guess = [1e3, 1e2, 1e4]
    popt, pcov = sp.optimize.curve_fit(modified_arrhenius, T, k, p0=initial_guess)
    A, B, E = popt
    print("A = ", A)
    print("B = ", B)
    print("E = ", E)

    plt.scatter(T, k, label='Data')
    t_fit = np.linspace(min(T), max(T), 1000)
    k_fit = modified_arrhenius(t_fit, A, B, E)
    plt.plot(t_fit, k_fit, label='Fit')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Rate constant (K)')
    plt.legend()
    plt.show()

def Problem2():
    """
    A wire carrying an electric current is surrounded by rubber insulation of outer radius r.
    The resistance of the wire generates heat, which is conducted through the insulation and convected into the surrounding air.
    The temperature of the wire can be shown to be: T = q/2*pi (ln(r/a)/k + 1/(hr)) + T_inf
    where q is heate generation (50 W/m),
    a is the wire radius (5 mm),
    k is the thermal conductivity of the insulation (0.16 W/m-K),
    h is the heat transfer coefficient (20 W/m^2-K),
    T_inf is the ambient temperature (280 K),
    and r is the outer radius of the insulation.
    """
    r = sympy.symbols('r', positive=True)
    q = 50
    a = 5e-3
    k = 0.16
    h = 20
    T_inf = 280
    T = q/(2*np.pi)*(sympy.ln(r/a)/k + 1/(h*r)) + T_inf
    dT_dr = sympy.diff(T, r)
    r_solution = sympy.solve(dT_dr, r)
    r_solution_eval = [sol.evalf() for sol in r_solution if sol > 0]
    print("Optimal outer radius of insulation")
    print(r_solution_eval)
    """
    Since function T is differentiable, simple differentiation using sympy can be used to find the optimal r value for the lowest temp.
    """

def Problem3():
    """
    A cable fixed at the ends carry the weights W1 and W2.
    The potential energy of the system is: V = -W1*y1 - W2*y2
    The principle of minimum potential energy states that the equilibrium configuration of the system is the one
    that satisfies geometric constraints and minimizes the potential energy.
    a) Write out the objective function to be minimized in terms of theta.
    y1 = L1*sin(theta1)
    y2 = L1*sin(theta1) + L2*sin(theta2)
    V = -W1*L1*sin(theta1) - W2*(L1*sin(theta1) + L2*sin(theta2))
    b) Write out the constraints given the geometry of B and H
    B = B1 + B2 + B3
    B__x = L__x*cos(theta__x)
    therefore B = L1*cos(theta1) + L2*cos(theta2) + L3*cos(theta3)
    H = H1 + H2 + H3
    H__y = L__y*sin(theta__y)
    therefore H = L1*sin(theta1) + L2*sin(theta2) + L3*sin(theta3)
    c) Determine equilibrium values for theta1, theta2, and theta3
    L__1 = 1.2 m
    L__2 = 1.5 m
    L__3 = 1.0 m
    B = 3.5 m
    H = 0 m
    W1 = 20e3 N
    W2 = 30e3 N
    """
    L__1 = 1.2
    L__2 = 1.5
    L__3 = 1.0
    B = 3.5
    H = 0
    W1 = 20e3
    W2 = 30e3
    def objective(angles):
        theta1, theta2, theta3 = angles
        return -W1*L__1*np.sin(theta1) - W2*(L__1*np.sin(theta1) + L__2*np.sin(theta2))
    def horizontal_constraint(angles):
        theta1, theta2, theta3 = angles
        return L__1*np.cos(theta1) + L__2*np.cos(theta2) + L__3*np.cos(theta3) - B
    def vertical_constraint(angles):
        theta1, theta2, theta3 = angles
        return L__1*np.sin(theta1) + L__2*np.sin(theta2) + L__3*np.sin(theta3) - H
    initial_guess = [0, 0, 0]
    constraints = [{'type': 'eq', 'fun': horizontal_constraint},
                   {'type': 'eq', 'fun': vertical_constraint}]
    result = sp.optimize.minimize(objective, initial_guess, constraints=constraints)
    print("Optimal angles (in radians)")
    print(result.x)


if __name__ == '__main__':
    Problem1()
    Problem2()
    Problem3()
