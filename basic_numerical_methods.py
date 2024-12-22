import numpy as np

def euler_method(f, t0, y0, h, n):
    """
    Euler method for solving ODEs
    
    Parameters:
    f (function): The function defining the ODE dy/dt = f(t, y)
    t0 (float): Initial time
    y0 (float): Initial value of y
    h (float): Step size
    n (int): Number of steps
    
    Returns:
    tuple: t_values and y_values
    """
    if h <= 0:
        raise ValueError("Step size h must be positive")
    if n < 0:
        raise ValueError("Number of steps n must be nonnegative")
    
    t_values = [t0]
    y_values = [y0]

    for _ in range(n):
        t = t_values[-1]
        y = y_values[-1]
        t_values.append(t + h)
        y_values.append(y + f(t, y)*(h))
        
    return t_values, y_values

def improved_euler_method(f, t0, y0, h, n):
    """
    Improved Euler method for solving ODEs
    
    Parameters:
    f (function): The function defining the ODE dy/dt = f(t, y)
    t0 (float): Initial time
    y0 (float): Initial value of y
    h (float): Step size
    n (int): Number of steps
    
    Returns:
    tuple: t_values and y_values
    """
    if h <= 0:
        raise ValueError("Step size h must be positive")
    if n < 0:
        raise ValueError("Number of steps n must be nonnegative")
    
    t_values = [t0]
    y_values = [y0]

    for _ in range(n):
        t = t_values[-1]
        y = y_values[-1]
        t_values.append(t + h)
        y_values.append(y + h / 2 * (f(t, y) + f(t + h, y + f(t, y) * h)))

    return t_values, y_values

def runge_kutta(f, t0, y0, h, n):
    """
    Runge-Kutta method for solving ODEs
    
    Parameters:
    f (function): The function defining the ODE dy/dt = f(t, y)
    t0 (float): Initial time
    y0 (float): Initial value of y
    h (float): Step size
    n (int): Number of steps
    
    Returns:
    tuple: t_values and y_values
    """
    if h <= 0:
        raise ValueError("Step size h must be positive")
    if n < 0:
        raise ValueError("Number of steps n must be nonnegative")
    
    t_values = [t0]
    y_values = [y0]

    for _ in range(n):
        t_n = t_values[-1]
        y_n = y_values[-1]
    
        k_n1 = f(t_n, y_n)
        k_n2 = f(t_n + .5 * h, y_n + .5 * h * k_n1)
        k_n3 = f(t_n + .5 * h, y_n + .5 * h * k_n2)
        k_n4 = f(t_n + h, y_n + h * k_n3)

        t_values.append(t_n + h)
        y_values.append(y_n + h * (k_n1 + 2 * k_n2 + 2 * k_n3 + k_n4) / 6)

    return t_values, y_values

def adams_bashforth_step(y_previous, last_four, counter):
    """ Adams-Bashforth step for computing the next value of y """
    return y_previous + h / 24 * (
            55 * last_four[(counter - 1) % 4]
            - 59 * last_four[(counter - 2) % 4]
            + 37 * last_four[(counter - 3) % 4]
            - 9 * last_four[counter]
            )

def adams_bashforth(f, t0, y0, h, n):
    """
    Adams-Bashforth method for solving ODEs
    
    Parameters:
    f (function): The function defining the ODE dy/dt = f(t, y)
    t0 (float): Initial time
    y0 (float): Initial value of y
    h (float): Step size
    n (int): Number of steps
    
    Returns:
    tuple: t_values and y_values
    """
    if h <= 0:
        raise ValueError("Step size h must be positive")
    if n < 3:
        raise ValueError("Number of steps n must be greater than or equal to 3")
    
    # Runge-Kutta method to get the first four values
    t_values, y_values = runge_kutta(f, t0, y0, h, 3)
    last_four = [f(t_values[i], y_values[i]) for i in range(4)]
    counter = 0
    
    for _ in range(n - 3):
        y_values.append(adams_bashforth_step(y_values[-1], last_four, counter))
        t_values.append(t_values[-1] + h)
        last_four[counter] = f(t_values[-1], y_values[-1])
        counter = (counter + 1) % 4
    
    return t_values, y_values

def predictor_corrector(f, t0, y0, h, n, max_iterations, tolerance):
    """
    Predictor-Corrector method using Adams-Bashforth and Adams-Moulton methods.
    
    Parameters:
    f (function): The function defining the ODE dy/dt = f(t, y)
    t0 (float): Initial time
    y0 (float): Initial value of y
    h (float): Step size
    n (int): Number of steps
    max_iterations (int): Maximum number of iterations for the corrector step
    tolerance (float): Tolerance for the corrector step
    
    Returns:
    tuple: t_values and y_values
    """
    if h <= 0:
        raise ValueError("Step size h must be positive")
    if n < 3:
        raise ValueError("Number of steps n must be greater than or equal to 3")
    
    # Runge-Kutta method to get the first four values
    t_values, y_values = runge_kutta(f, t0, y0, h, 3)
    last_four = [f(t_values[i], y_values[i]) for i in range(4)]
    counter = 0
    
    for _ in range(n - 3):
        t_next = t_values[-1] + h
        y_previous = y_values[-1]
        y_predicted = adams_bashforth_step(y_previous, last_four, counter)

        # Iterative corrector step
        for _ in range(max_iterations):
            y_corrected = y_previous + h / 24 * (
                9 * f(t_next, y_predicted)
                + 19 * last_four[(counter - 1) % 4]
                - 5 * last_four[(counter - 2) % 4]
                + last_four[(counter - 3) % 4]
            )
            if np.abs(y_corrected - y_predicted) < tolerance:
                break
            y_predicted = y_corrected

        t_values.append(t_next)
        y_values.append(y_corrected)
        last_four[counter] = f(t_next, y_corrected)
        counter = (counter + 1) % 4

    return t_values, y_values

def f(t, y):
    # The function defining the ODE dy/dt = f(t,y)
    return 1 - t + 4 * y

def analytical_solution(t):
    # The analytical solution to the ODE for error calculation
    return (.25 * t) - 3/16 + (19/16 * np.exp(4 * t))

def calculate_error(y_numerical, y_exact):
    # Calculate the error between the numerical and exact solutions
    return np.abs(y_numerical - y_exact)

def mean_absolute_error(y_numerical, y_exact):
    # Calculate the mean absolute error between the numerical and exact solutions
    return np.mean(calculate_error(y_numerical, y_exact))

def root_mean_square_error(y_numerical, y_exact):
    # Calculate the root mean square error between the numerical and exact solutions
    return np.sqrt(np.mean(calculate_error(y_numerical, y_exact) ** 2))

if __name__ == "__main__":
    # Initial conditions and parameters
    t0 = 0
    y0 = 1
    h = .00001
    n = 100000

    # Numerical methods
    t_euler, y_euler = euler_method(f, t0, y0, h, n)
    t_improved_euler, y_improved_euler = improved_euler_method(f, t0, y0, h, n)
    t_runge_kutta, y_runge_kutta = runge_kutta(f, t0, y0, h, n)
    t_adams_bashforth, y_adams_bashforth = adams_bashforth(f, t0, y0, h, n)
    t_predictor_corrector, y_predictor_corrector = predictor_corrector(f, t0, y0, h, n, 3, 1e-6)

    # Analytical solution
    t_exact = np.linspace(t0, t0 + h * n, n + 1)
    y_exact = analytical_solution(t_exact)

    # Errors
    mae_euler = mean_absolute_error(y_euler, y_exact)
    rmse_euler = root_mean_square_error(y_euler, y_exact)

    mae_improved_euler = mean_absolute_error(y_improved_euler, y_exact)
    rmse_improved_euler = root_mean_square_error(y_improved_euler, y_exact)

    mae_runge_kutta = mean_absolute_error(y_runge_kutta, y_exact)
    rmse_runge_kutta = root_mean_square_error(y_runge_kutta, y_exact)

    mae_adams_bashforth = mean_absolute_error(y_adams_bashforth, y_exact)
    rmse_adams_bashforth = root_mean_square_error(y_adams_bashforth, y_exact)

    mae_predictor_corrector = mean_absolute_error(y_predictor_corrector, y_exact)
    rmse_predictor_corrector = root_mean_square_error(y_predictor_corrector, y_exact)

    # Print errors
    print(f"Euler Method - MAE: {mae_euler}, RMSE: {rmse_euler}")
    print(f"Improved Euler Method - MAE: {mae_improved_euler}, RMSE: {rmse_improved_euler}")
    print(f"Runge-Kutta Method - MAE: {mae_runge_kutta}, RMSE: {rmse_runge_kutta}")
    print(f"Adams-Bashforth Method - MAE: {mae_adams_bashforth}, RMSE: {rmse_adams_bashforth}")
    print(f"Predictor-Corrector Method - MAE: {mae_predictor_corrector}, RMSE: {rmse_predictor_corrector}")