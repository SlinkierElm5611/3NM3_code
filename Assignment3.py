import numpy as np
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt

def Problem1():
    """
    A nuclear fuel pellet is a cylinder, 1.5 cm in lenth and 1 cm in diameter.
    Assume the surface temperature is 300 C everywhere. Given temperature probe data below,
    determine the radial temperature profile in the middle
    of a nuclear fuel pellet (i.e.: T(r, z = 0.75))
    using radial basis functions.
    Consider what you know about this system. What extra information do you have in terms of
    a) type(s) of symmetry?
    b) boundary conditions?
    c) Plot the best guess of the radial temperature gradient.
    """
    # 20 data points presented in columns: | x | y | z | T |
    data = np.array([
        [5.1690e-02, 2.3766e-01, 6.7059e-01, 5.2645e+02],
        [1.1353e-01, 9.4708e-02, 5.3856e-01, 5.5201e+02],
        [1.6676e-01, 1.4358e-01, 4.6936e-01, 5.0802e+02],
        [1.3610e-01, 3.7207e-02, 2.1694e-01, 4.3663e+02],
        [8.9225e-02, 3.7293e-01, 1.1270e+00, 3.9234e+02],
        [1.9001e-01, 3.7240e-01, 8.4774e-01, 3.8872e+02],
        [5.4849e-02, 3.5425e-01, 5.7478e-01, 4.3784e+02],
        [1.7001e-01, 2.0241e-01, 1.2960e+00, 4.0159e+02],
        [2.0606e-01, 3.1594e-01, 6.4077e-01, 4.2652e+02],
        [2.5382e-01, 2.5859e-01, 4.8610e-01, 4.2481e+02],
        [5.6038e-02, 8.2231e-02, 4.2029e-01, 5.3244e+02],
        [3.1242e-01, 8.0489e-02, 1.1530e+00, 4.2453e+02],
        [6.0186e-02, 4.4891e-01, 3.9941e-01, 3.4207e+02],
        [1.5070e-01, 3.4794e-01, 1.5595e-01, 3.4750e+02],
        [1.8215e-01, 3.4388e-01, 1.0478e+00, 3.9963e+02],
        [1.1633e-01, 4.1011e-01, 5.5001e-01, 3.7611e+02],
        [1.2377e-01, 3.3703e-01, 3.7672e-02, 3.1423e+02],
        [4.6378e-02, 3.3653e-01, 1.4434e+00, 3.2345e+02],
        [2.9063e-02, 3.2584e-02, 2.3977e-01, 4.5993e+02],
        [2.1162e-02, 3.8590e-01, 2.5905e-01, 3.6901e+02]
    ])
    """
    a)
    We have symmetry around the z-axis as the cylinder is symmetric about the z-axis.
    There is also symmetry around the x-axis and the y-axis for the same reasons.
    """
    """
    b)
    The boundary conditions are that the temperature is 300 C everywhere on the surface of the cylinder.
    """
    #c)
    x,y,z,T = data[:,0], data[:,1], data[:,2], data[:,3]

    r = np.sqrt(x**2 + y**2)

    rbf = Rbf(r, z, T, function='multiquadric', smooth=0)

    r_vals = np.linspace(0, 0.5, 100)
    z_mid = 0.75

    T_r = rbf(r_vals, np.full_like(r_vals, z_mid))

    T_r [r_vals > 0.5] = 300

    plt.plot(r_vals, T_r)
    plt.xlabel("r")
    plt.ylabel("T")
    plt.title("Radial Temperature Profile")
    plt.show()


def Problem2():
    """
    You run an experiment and obtain the following data
    """
    # data in form of |x|y1|y2|y3|y4|y5|
    d = np.array([
    [0.00, -29.49, -2.14, 15.88, 22.69, 28.53],
    [1.11, 2.83, 18.02, -25.45, -32.45, 7.50],
    [2.22, 1.97, -10.49, -0.18, -32.10, -40.31],
    [3.33, -38.09, -46.16, -7.87, -33.97, -38.39],
    [4.44, -3.97, -32.22, -33.95, -11.07, -32.47],
    [5.56, 4.45, -10.88, 20.43, 6.57, -8.49],
    [6.67, 50.22, 51.29, 80.02, 66.15, 84.90],
    [7.78, 164.11, 190.26, 160.94, 182.35, 163.18],
    [8.89, 331.75, 306.51, 278.40, 302.13, 335.44],
    [10.00, 517.06, 483.20, 476.73, 512.16, 500.64]
    ])
    #a)Determine the best cubic polynomial fit to this data with the uncertainty
    x = d[:,0]
    y_values = d[:,1:]
    y_mean = np.mean(y_values, axis=1)
    y_std = np.std(y_values, axis=1, ddof=1)
    weights = 1/y_std**2
    coeffs, cov = np.polyfit(x, y_mean, deg=3, w=weights, cov=True)
    p = np.poly1d(coeffs)
    coeffs_std = np.sqrt(np.diag(cov))
    print("Coefficients for cubic polynomial fit: ", coeffs)
    print("Standard deviations for coefficients: ", coeffs_std)
    #b) Your manager thinks this should be a quadratic. Which do you think it should be and why?
    coeefs_quad, cov_quad = np.polyfit(x, y_mean, deg=2, w=weights, cov=True)
    p_quad = np.poly1d(coeefs_quad)
    coeffs_std_quad = np.sqrt(np.diag(cov_quad))
    print("Coefficients for quadratic polynomial fit: ", coeefs_quad)
    print("Standard deviations for coefficients: ", coeffs_std_quad)

    x_fit = np.linspace(min(x), max(x), 200)
    y_fit = p(x_fit)
    y_fit_quad = p_quad(x_fit)

    rss_cubic = np.sum(((y_mean - p(x)) * weights)**2)
    rss_quad = np.sum(((y_mean - p_quad(x)) * weights)**2)

    def adjusted_r_squared(y_obs, y_pred, p):
        n = len(y_obs)
        residual = y_obs - y_pred
        ss_res = np.sum(residual**2)
        ss_tot = np.sum((y_obs - np.mean(y_obs))**2)
        r2 = 1 - (ss_res / ss_tot)
        r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        return r2_adj

    r2_adj_cubic = adjusted_r_squared(y_mean, p(x), 3)
    r2_adj_quad = adjusted_r_squared(y_mean, p_quad(x), 2)

    print(f"\nComparison Metrics:")
    print(f"Cubic Model RSS: {rss_cubic:.2f}, Adjusted R²: {r2_adj_cubic:.4f}")
    print(f"Quadratic Model RSS: {rss_quad:.2f}, Adjusted R²: {r2_adj_quad:.4f}")

    plt.figure(figsize=(10, 6))
    plt.errorbar(x, y_mean, yerr=y_std, fmt='o', label='Experimental Data', capsize=5)
    plt.plot(x_fit, y_fit, 'r-', label='Cubic Fit')
    plt.plot(x_fit, y_fit_quad, 'g--', label='Quadratic Fit')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Polynomial Fits to Experimental Data')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    Problem1()
    Problem2()
