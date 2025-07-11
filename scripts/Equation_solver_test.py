from sympy import symbols, Eq, solve, I
import math

x, x0, y0, R = symbols('x x0 y0 R')

# setting up the complex equation z^2 + 1 = 0
equation = Eq(x**2*(x0**2+y0**2) - 2*R**2*x*x0 + R**2*(R**2-y0**2), 0)

# solving the equation symbolically to find complex solutions
solutions = solve(equation, x)
# printing solutions
print(solutions)