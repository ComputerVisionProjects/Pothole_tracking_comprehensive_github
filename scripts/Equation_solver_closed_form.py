from math import sqrt
import time
import random

def tangent_point_finder(x0, y0, R):
        start = time.perf_counter()
        denominator = x0**2 + y0**2
        if denominator == 0:
            raise ValueError("Denominator cannot be zero (x0 and y0 cannot both be zero).")
        
        term_inside_sqrt = R**4 * x0**2 - R**2 * (x0**2 + y0**2) * (R**2 - y0**2)
        
        if term_inside_sqrt < 0:
            raise ValueError("Negative value under the square root. No real solution.")
        
        sqrt_term = sqrt(term_inside_sqrt)
        
        x1 = (R**2 * x0 + sqrt_term) / denominator
        x2 = (R**2 * x0 - sqrt_term) / denominator

        if x0 > 0:
            x_t = x2

        if x0 < 0:
            x_t = x1

        y_t = sqrt(R**2 - x_t**2)

        end = time.perf_counter()
        inference_time = end - start
        print(f"inference time: {inference_time}")
        return [x_t, y_t]



