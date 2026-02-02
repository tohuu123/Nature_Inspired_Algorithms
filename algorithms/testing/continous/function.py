# functions to test optimization algorithms 
import numpy as np 
from numpy import asarray

# sphere functionn 
def sphere(x): 
    x = asarray(x)
    return sum(x ** 2)

# Rastrigin function (highly multimodal)
def rastrigin(x):
    x = asarray(x)
    n = len(x)
    return 10 * n + sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

# Rosenbrock function (narrow valley)
def rosenbrock(x):
    x = asarray(x)
    return sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

# Griewank function (regularly distributed minima)
def griewank(x):
    x = asarray(x)
    sum_term = sum(x ** 2) / 4000
    prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return 1 + sum_term - prod_term

# Ackley function (many local optima)
def ackley(x):
    x = asarray(x)
    n = len(x)
    sum_sq = sum(x ** 2)
    sum_cos = sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + 20 + np.e