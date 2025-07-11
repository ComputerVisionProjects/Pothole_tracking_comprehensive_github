# importing from sympy library
from sympy import symbols, Eq, solve, I
import math
import time

def tangent_point_finder(x_0, y_0, Radius):
    # defining the symbolic variable 'z'
    start = time.perf_counter()
    x = symbols('x')

    # setting up the complex equation z^2 + 1 = 0
    equation = Eq(x**2*(x_0**2+y_0**2) - 2*Radius**2*x*x_0 + Radius**2*(Radius**2-y_0**2), 0)

    # solving the equation symbolically to find complex solutions
    solutions = solve(equation, x)
    if  x_0 > 0:
        x_t = solutions[0]

    else:
        x_t = solutions[1]

    y_t = math.sqrt(Radius**2 - x_t**2)
    end = time.perf_counter()
    inference_time = end - start
    print(f"inference time: {inference_time}")
    return [float(x_t), float(y_t)]

x = 30
y = 40
tangent_point_finder(x, y, 33.1)